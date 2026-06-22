const NNUE_INPUT   = 768
const NNUE_HL      = 1024
const NNUE_HL2     = NNUE_HL ÷ 2  # 512 — width after l0's crelu+pairwise_mul
const NNUE_L1      = 16           # l1 output width per bucket
const NNUE_L2      = 32           # l2 output width per bucket
const NNUE_BUCKETS = 8
const NNUE_SCALE   = 400
const NNUE_QA      = 255
const NNUE_QB      = 64
const NNUE_IB           = 16   # king input buckets 
const NNUE_FINNY_SLOTS  = 2 * NNUE_IB  # mirror (0/1) × bucket (0..15) = 32 slots per perspective

# King bucket layout: 32 entries indexed by rank*4 + mirror_file
# rank=0(rank1)..7(rank8), mirror_file = min(file, 7-file)
const _IB_LAYOUT = (
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
    14, 14, 15, 15,
)

# King bucket from standard 0-based square (A1=0, ..., H8=63)
@inline function _king_bucket(sq::Int)::Int
    rank  = sq >> 3
    file  = sq & 7
    mfile = file > 3 ? 7 - file : file
    @inbounds _IB_LAYOUT[rank * 4 + mfile + 1]
end

# File-flip value: XOR feature index with 7 when king is on files E-H
@inline _king_flip(sq::Int) = (sq & 7) > 3 ? 7 : 0

struct NNUENet
    l0w::Matrix{Int16}      # NNUE_HL × (NNUE_INPUT × NNUE_IB)
    l0b::Vector{Int16}      # NNUE_HL
    l1w::Array{Float32,3}   # NNUE_HL × NNUE_L1 × NNUE_BUCKETS (dequantised from Int8, QB)
    l1b::Matrix{Float32}    # NNUE_L1 × NNUE_BUCKETS
    l2w::Array{Float32,3}   # NNUE_L1 × NNUE_L2 × NNUE_BUCKETS
    l2b::Matrix{Float32}    # NNUE_L2 × NNUE_BUCKETS
    l3w::Array{Float32,3}   # NNUE_L2 × 1 × NNUE_BUCKETS
    l3b::Matrix{Float32}    # 1 × NNUE_BUCKETS
end

# quantised.bin layout (from trainer/src/train_ml.rs). 
# See misc for a visualization
function load_nnue(path::String)::NNUENet
    open(path, "r") do io
        l0w = Matrix{Int16}(undef, NNUE_HL, NNUE_INPUT * NNUE_IB)
        read!(io, l0w)
        l0b = Vector{Int16}(undef, NNUE_HL)
        read!(io, l0b)

        l1w_raw = Matrix{Int8}(undef, NNUE_HL, NNUE_BUCKETS * NNUE_L1)
        read!(io, l1w_raw)
        l1w = reshape(Float32.(l1w_raw) ./ Float32(NNUE_QB), NNUE_HL, NNUE_L1, NNUE_BUCKETS)

        l1b_raw = Vector{Float32}(undef, NNUE_BUCKETS * NNUE_L1)
        read!(io, l1b_raw)
        l1b = reshape(l1b_raw, NNUE_L1, NNUE_BUCKETS)

        l2w_raw = Matrix{Float32}(undef, NNUE_L1, NNUE_BUCKETS * NNUE_L2)
        read!(io, l2w_raw)
        l2w = reshape(l2w_raw, NNUE_L1, NNUE_L2, NNUE_BUCKETS)

        l2b_raw = Vector{Float32}(undef, NNUE_BUCKETS * NNUE_L2)
        read!(io, l2b_raw)
        l2b = reshape(l2b_raw, NNUE_L2, NNUE_BUCKETS)

        l3w_raw = Matrix{Float32}(undef, NNUE_L2, NNUE_BUCKETS)
        read!(io, l3w_raw)
        l3w = reshape(l3w_raw, NNUE_L2, 1, NNUE_BUCKETS)

        l3b_raw = Vector{Float32}(undef, NNUE_BUCKETS)
        read!(io, l3b_raw)
        l3b = reshape(l3b_raw, 1, NNUE_BUCKETS)

        NNUENet(l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b)
    end
end

# MaterialCount<NNUE_BUCKETS>: (non-king pieces) / ceil(32/NNUE_BUCKETS), 1-indexed
function _material_bucket(b::Board)::Int
    n = count_ones(b.bb[1] | b.bb[2] | b.bb[3] | b.bb[4] | b.bb[5] |
                   b.bb[7] | b.bb[8] | b.bb[9] | b.bb[10] | b.bb[11])
    n ÷ cld(32, NNUE_BUCKETS) + 1
end

# Finny cache entry: one per (perspective × mirror × bucket) slot.
# Caches an accumulator together with the piece bitboards used to build it.
# On refresh, only the diff vs. the cached board state needs to be applied.
mutable struct FinnyEntry
    acc::Vector{Int32}   # NNUE_HL cached accumulator
    bb::Vector{UInt64}   # 12 piece bitboards matching b.bb[1..12]
    valid::Bool
end
FinnyEntry() = FinnyEntry(zeros(Int32, NNUE_HL), zeros(UInt64, 12), false)

# Finny slot index (1-based, range 1..NNUE_FINNY_SLOTS=32)
# White perspective: bucket from physical wksq, mirror from physical wksq file
@inline _finny_slot_w(wksq::Int) =
    ((wksq & 7) >= 4 ? NNUE_IB : 0) + _king_bucket(wksq) + 1

# Black perspective: bucket from rank-flipped bksq, mirror from physical bksq file
@inline _finny_slot_b(bksq::Int) =
    ((bksq & 7) >= 4 ? NNUE_IB : 0) + _king_bucket(bksq ⊻ 56) + 1

# Two NNUE_HL accumulators (white / black perspective) plus king state and Finny caches.
# stm_hidden/ntm_hidden/hl1/hl2/hl3/out are scratch buffers for nnue_eval's forward
mutable struct Accumulator
    w::Vector{Int32}
    b::Vector{Int32}
    stm_hidden::Vector{Float32}   
    ntm_hidden::Vector{Float32}   
    hl1::Vector{Float32}          
    hl2::Vector{Float32}          
    hl3::Vector{Float32}         
    out::Vector{Float32}         
    wksq::Int    # white king std 0-based square
    bksq::Int    # black king std 0-based square
    dirty::Bool  # true ⟹ needs refresh! before eval
    finny_w::Vector{FinnyEntry}   # NNUE_FINNY_SLOTS entries for white perspective
    finny_b::Vector{FinnyEntry}   # NNUE_FINNY_SLOTS entries for black perspective
end

Accumulator() = Accumulator(
    zeros(Int32, NNUE_HL), zeros(Int32, NNUE_HL),
    zeros(Float32, NNUE_HL2), zeros(Float32, NNUE_HL2),
    zeros(Float32, NNUE_HL), zeros(Float32, NNUE_L1),
    zeros(Float32, NNUE_L2), zeros(Float32, 1),
    4, 60, true,
    [FinnyEntry() for _ in 1:NNUE_FINNY_SLOTS],
    [FinnyEntry() for _ in 1:NNUE_FINNY_SLOTS],
)

# sq.val is 1-indexed (A1=1 … H8=64); 0-based standard (A1=0 … H8=63)
@inline _nnue_sq(v::Int) = v - 1

# 1-indexed column in fw: bucket offset + (side*384 + pt*64 + sq) XOR file-flip
@inline _feat_ib(bkt::Int, flip::Int, side::Int, pt::Int, sq::Int) =
    (side * 384 + pt * 64 + sq) ⊻ flip + bkt * NNUE_INPUT + 1

@inline function _acc_add!(v::Vector{Int32}, fw::Matrix{Int16}, bkt::Int, flip::Int, side::Int, pt::Int, sq::Int)
    f = _feat_ib(bkt, flip, side, pt, sq)
    @inbounds @simd for j in 1:NNUE_HL
        v[j] += Int32(fw[j, f])
    end
end

@inline function _acc_sub!(v::Vector{Int32}, fw::Matrix{Int16}, bkt::Int, flip::Int, side::Int, pt::Int, sq::Int)
    f = _feat_ib(bkt, flip, side, pt, sq)
    @inbounds @simd for j in 1:NNUE_HL
        v[j] -= Int32(fw[j, f])
    end
end

# Finny table
# Diffs b.bb vs entry.bb, applies only the changed pieces, then copies to dst.
# black_pov=true  ⟹ rank-flip squares (⊻56) and swap own/opp sides.
# On first call, entry.bb is all zeros, so diff = full board.
function _refresh_finny!(dst::Vector{Int32}, fw::Matrix{Int16}, fb::Vector{Int16},
                         entry::FinnyEntry, bkt::Int, flip::Int, b::Board, black_pov::Bool)
    if !entry.valid
        @inbounds for i in 1:NNUE_HL; entry.acc[i] = Int32(fb[i]); end
        entry.valid = true
    end

    for (pt_jl, pt) in ((PAWN,0),(KNIGHT,1),(BISHOP,2),(ROOK,3),(QUEEN,4),(KING,5))
        # white pieces (bb index pt_jl): own=0 for white pov, opp=1 for black pov
        let side = black_pov ? 1 : 0
            δa = b.bb[pt_jl]     & ~entry.bb[pt_jl]
            δr = entry.bb[pt_jl] & ~b.bb[pt_jl]
            entry.bb[pt_jl] = b.bb[pt_jl]
            while δa != 0
                sq = Int(trailing_zeros(δa)); δa &= δa - 1
                _acc_add!(entry.acc, fw, bkt, flip, side, pt, black_pov ? sq ⊻ 56 : sq)
            end
            while δr != 0
                sq = Int(trailing_zeros(δr)); δr &= δr - 1
                _acc_sub!(entry.acc, fw, bkt, flip, side, pt, black_pov ? sq ⊻ 56 : sq)
            end
        end
        # black pieces (bb index pt_jl+6): opp=1 for white pov, own=0 for black pov
        let side = black_pov ? 0 : 1
            δa = b.bb[pt_jl+6]     & ~entry.bb[pt_jl+6]
            δr = entry.bb[pt_jl+6] & ~b.bb[pt_jl+6]
            entry.bb[pt_jl+6] = b.bb[pt_jl+6]
            while δa != 0
                sq = Int(trailing_zeros(δa)); δa &= δa - 1
                _acc_add!(entry.acc, fw, bkt, flip, side, pt, black_pov ? sq ⊻ 56 : sq)
            end
            while δr != 0
                sq = Int(trailing_zeros(δr)); δr &= δr - 1
                _acc_sub!(entry.acc, fw, bkt, flip, side, pt, black_pov ? sq ⊻ 56 : sq)
            end
        end
    end
    copyto!(dst, entry.acc)
end

# crelu(x) = clamp(x, 0, QA), then split into 2 halves and multiply element-wise.
# This is for processing l0 before multiplying with the weight matrix to form l1
@inline function _crelu_pairwise_mul!(out::Vector{Float32}, acc::Vector{Int32})
    half = length(acc) ÷ 2
    inv_qa2 = 1f0 / Float32(NNUE_QA)^2
    @inbounds @simd for i in 1 : half
        x = clamp(acc[i], Int32(0), Int32(NNUE_QA))
        y = clamp(acc[half + i], Int32(0), Int32(NNUE_QA))
        out[i] = Float32(x * y) * inv_qa2
    end
end

# screlu function
@inline function _screlu(x::Float32)::Float32
    c = clamp(x, 0f0, 1f0)
    return c * c
end

# Compute the next layer in the feedforward network. Shared by l1, l2, l3 — all
# operate on plain Float32 (l1w/l1b are dequantised to Float32 once in load_nnue,
# so no quantisation handling is needed here).
@inline function _dense_forward!(
    out::AbstractVector{Float32},
    input::AbstractVector{Float32},
    w::AbstractArray{Float32,3},
    b::AbstractMatrix{Float32},
    bkt::Int,
    activation::Bool)

    d_in, d_out = size(w, 1), size(w, 2)
    @inbounds for j in 1:d_out
        s = b[j, bkt]
        @simd for i in 1:d_in
            s += input[i] * w[i, j, bkt]
        end
        out[j] = activation ? _screlu(s) : s
    end
end

# Refresh both perspectives via the Finny cache. Called on king moves and at search root.
function refresh!(acc::Accumulator, b::Board, net::NNUENet)
    wksq  = Int(trailing_zeros(b.bb[BB_WK]))
    bksq  = Int(trailing_zeros(b.bb[BB_BK]))
    wbkt  = _king_bucket(wksq);       wflip = _king_flip(wksq)
    bbkt  = _king_bucket(bksq ⊻ 56); bflip = _king_flip(bksq)
    wslot = _finny_slot_w(wksq)
    bslot = _finny_slot_b(bksq)
    _refresh_finny!(acc.w, net.l0w, net.l0b, acc.finny_w[wslot], wbkt, wflip, b, false)
    _refresh_finny!(acc.b, net.l0w, net.l0b, acc.finny_b[bslot], bbkt, bflip, b, true)
    acc.wksq  = wksq
    acc.bksq  = bksq
    acc.dirty = false
end

# Incremental update for a regular or capture move (pre-move board state)
@inline function _apply_move!(acc::Accumulator, b::Board, m::UInt64, fw::Matrix{Int16},
                               wbkt::Int, wflip::Int, bbkt::Int, bflip::Int, ::Val{add}) where {add}
    piece = pieceon(b, from(m))
    pt    = ptype(piece).val - 1
    fsq   = _nnue_sq(from(m).val)
    tsq   = _nnue_sq(to(m).val)

    if pcolor(piece) == WHITE
        fw_to  = _feat_ib(wbkt, wflip, 0, pt, tsq)
        fw_frm = _feat_ib(wbkt, wflip, 0, pt, fsq)
        fb_to  = _feat_ib(bbkt, bflip, 1, pt, tsq ⊻ 56)
        fb_frm = _feat_ib(bbkt, bflip, 1, pt, fsq ⊻ 56)
        if moveiscapture(b, m)
            cp     = ptype(pieceon(b, to(m))).val - 1
            fw_cap = _feat_ib(wbkt, wflip, 1, cp, tsq)
            fb_cap = _feat_ib(bbkt, bflip, 0, cp, tsq ⊻ 56)
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fw_to]) - Int32(fw[j, fw_frm]) - Int32(fw[j, fw_cap])
                acc.w[j] += add ? δ : -δ
            end
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fb_to]) - Int32(fw[j, fb_frm]) - Int32(fw[j, fb_cap])
                acc.b[j] += add ? δ : -δ
            end
        else
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fw_to]) - Int32(fw[j, fw_frm])
                acc.w[j] += add ? δ : -δ
            end
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fb_to]) - Int32(fw[j, fb_frm])
                acc.b[j] += add ? δ : -δ
            end
        end
    else
        fb_to  = _feat_ib(bbkt, bflip, 0, pt, tsq ⊻ 56)
        fb_frm = _feat_ib(bbkt, bflip, 0, pt, fsq ⊻ 56)
        fw_to  = _feat_ib(wbkt, wflip, 1, pt, tsq)
        fw_frm = _feat_ib(wbkt, wflip, 1, pt, fsq)
        if moveiscapture(b, m)
            cp     = ptype(pieceon(b, to(m))).val - 1
            fb_cap = _feat_ib(bbkt, bflip, 1, cp, tsq ⊻ 56)
            fw_cap = _feat_ib(wbkt, wflip, 0, cp, tsq)
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fb_to]) - Int32(fw[j, fb_frm]) - Int32(fw[j, fb_cap])
                acc.b[j] += add ? δ : -δ
            end
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fw_to]) - Int32(fw[j, fw_frm]) - Int32(fw[j, fw_cap])
                acc.w[j] += add ? δ : -δ
            end
        else
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fb_to]) - Int32(fw[j, fb_frm])
                acc.b[j] += add ? δ : -δ
            end
            @inbounds @simd for j in 1:NNUE_HL
                δ = Int32(fw[j, fw_to]) - Int32(fw[j, fw_frm])
                acc.w[j] += add ? δ : -δ
            end
        end
    end
end

@inline function _apply_ep!(acc::Accumulator, b::Board, m::UInt64, fw::Matrix{Int16},
                             wbkt::Int, wflip::Int, bbkt::Int, bflip::Int, ::Val{add}) where {add}
    fsq   = _nnue_sq(from(m).val)
    tsq   = _nnue_sq(to(m).val)
    ep_sq = ((from(m).val - 1) >> 3) * 8 + ((to(m).val - 1) & 7)

    if pcolor(pieceon(b, from(m))) == WHITE
        add ? _acc_sub!(acc.w, fw, wbkt, wflip, 0, 0, fsq)         : _acc_add!(acc.w, fw, wbkt, wflip, 0, 0, fsq)
        add ? _acc_add!(acc.w, fw, wbkt, wflip, 0, 0, tsq)         : _acc_sub!(acc.w, fw, wbkt, wflip, 0, 0, tsq)
        add ? _acc_sub!(acc.w, fw, wbkt, wflip, 1, 0, ep_sq)       : _acc_add!(acc.w, fw, wbkt, wflip, 1, 0, ep_sq)
        add ? _acc_sub!(acc.b, fw, bbkt, bflip, 1, 0, fsq ⊻ 56)   : _acc_add!(acc.b, fw, bbkt, bflip, 1, 0, fsq ⊻ 56)
        add ? _acc_add!(acc.b, fw, bbkt, bflip, 1, 0, tsq ⊻ 56)   : _acc_sub!(acc.b, fw, bbkt, bflip, 1, 0, tsq ⊻ 56)
        add ? _acc_sub!(acc.b, fw, bbkt, bflip, 0, 0, ep_sq ⊻ 56) : _acc_add!(acc.b, fw, bbkt, bflip, 0, 0, ep_sq ⊻ 56)
    else
        add ? _acc_sub!(acc.b, fw, bbkt, bflip, 0, 0, fsq ⊻ 56)   : _acc_add!(acc.b, fw, bbkt, bflip, 0, 0, fsq ⊻ 56)
        add ? _acc_add!(acc.b, fw, bbkt, bflip, 0, 0, tsq ⊻ 56)   : _acc_sub!(acc.b, fw, bbkt, bflip, 0, 0, tsq ⊻ 56)
        add ? _acc_sub!(acc.b, fw, bbkt, bflip, 1, 0, ep_sq ⊻ 56) : _acc_add!(acc.b, fw, bbkt, bflip, 1, 0, ep_sq ⊻ 56)
        add ? _acc_sub!(acc.w, fw, wbkt, wflip, 1, 0, fsq)         : _acc_add!(acc.w, fw, wbkt, wflip, 1, 0, fsq)
        add ? _acc_add!(acc.w, fw, wbkt, wflip, 1, 0, tsq)         : _acc_sub!(acc.w, fw, wbkt, wflip, 1, 0, tsq)
        add ? _acc_sub!(acc.w, fw, wbkt, wflip, 0, 0, ep_sq)       : _acc_add!(acc.w, fw, wbkt, wflip, 0, 0, ep_sq)
    end
end

@inline function _apply_promo!(acc::Accumulator, b::Board, m::UInt64, fw::Matrix{Int16},
                                wbkt::Int, wflip::Int, bbkt::Int, bflip::Int, ::Val{add}) where {add}
    fsq      = _nnue_sq(from(m).val)
    tsq      = _nnue_sq(to(m).val)
    promo_pt = promotion(m).val - 1

    if pcolor(pieceon(b, from(m))) == WHITE
        add ? _acc_sub!(acc.w, fw, wbkt, wflip, 0, 0, fsq)               : _acc_add!(acc.w, fw, wbkt, wflip, 0, 0, fsq)
        add ? _acc_sub!(acc.b, fw, bbkt, bflip, 1, 0, fsq ⊻ 56)         : _acc_add!(acc.b, fw, bbkt, bflip, 1, 0, fsq ⊻ 56)
        add ? _acc_add!(acc.w, fw, wbkt, wflip, 0, promo_pt, tsq)        : _acc_sub!(acc.w, fw, wbkt, wflip, 0, promo_pt, tsq)
        add ? _acc_add!(acc.b, fw, bbkt, bflip, 1, promo_pt, tsq ⊻ 56)  : _acc_sub!(acc.b, fw, bbkt, bflip, 1, promo_pt, tsq ⊻ 56)
        if moveiscapture(b, m)
            cp = ptype(pieceon(b, to(m))).val - 1
            add ? _acc_sub!(acc.w, fw, wbkt, wflip, 1, cp, tsq)        : _acc_add!(acc.w, fw, wbkt, wflip, 1, cp, tsq)
            add ? _acc_sub!(acc.b, fw, bbkt, bflip, 0, cp, tsq ⊻ 56)  : _acc_add!(acc.b, fw, bbkt, bflip, 0, cp, tsq ⊻ 56)
        end
    else
        add ? _acc_sub!(acc.b, fw, bbkt, bflip, 0, 0, fsq ⊻ 56)         : _acc_add!(acc.b, fw, bbkt, bflip, 0, 0, fsq ⊻ 56)
        add ? _acc_sub!(acc.w, fw, wbkt, wflip, 1, 0, fsq)               : _acc_add!(acc.w, fw, wbkt, wflip, 1, 0, fsq)
        add ? _acc_add!(acc.b, fw, bbkt, bflip, 0, promo_pt, tsq ⊻ 56)  : _acc_sub!(acc.b, fw, bbkt, bflip, 0, promo_pt, tsq ⊻ 56)
        add ? _acc_add!(acc.w, fw, wbkt, wflip, 1, promo_pt, tsq)        : _acc_sub!(acc.w, fw, wbkt, wflip, 1, promo_pt, tsq)
        if moveiscapture(b, m)
            cp = ptype(pieceon(b, to(m))).val - 1
            add ? _acc_sub!(acc.b, fw, bbkt, bflip, 1, cp, tsq ⊻ 56)  : _acc_add!(acc.b, fw, bbkt, bflip, 1, cp, tsq ⊻ 56)
            add ? _acc_sub!(acc.w, fw, wbkt, wflip, 0, cp, tsq)        : _acc_add!(acc.w, fw, wbkt, wflip, 0, cp, tsq)
        end
    end
end

# Incremental update. Call BEFORE domove!(b, m).
# King moves (including castling) set dirty flag; refresh! is deferred to nnue_eval.
function update!(acc::Accumulator, b::Board, m::UInt64, net::NNUENet)
    # king move (including castling) shifts the bucket, all feature offsets invalid;
    # defer full recompute to nnue_eval, which has the post-move board
    ptype(pieceon(b, from(m))) == KING && (acc.dirty = true; return)

    # non-king move: bucket/flip are stable, read from stored king squares
    wbkt  = _king_bucket(acc.wksq);       wflip = _king_flip(acc.wksq)
    bbkt  = _king_bucket(acc.bksq ⊻ 56); bflip = _king_flip(acc.bksq)
    fw    = net.l0w
    mflag = Int((m >> 16) & 0xf)

    if mflag == 4
        # en-passant
        _apply_ep!(acc, b, m, fw, wbkt, wflip, bbkt, bflip, Val(true))
    elseif promotion(m) ≠ PieceType(0)
        # promotion
        _apply_promo!(acc, b, m, fw, wbkt, wflip, bbkt, bflip, Val(true))
    else
        _apply_move!(acc, b, m, fw, wbkt, wflip, bbkt, bflip, Val(true))
    end
end

# Reverse the update. Call AFTER undomove!(b, u) — board is back to pre-move state.
function undo_update!(acc::Accumulator, b::Board, m::UInt64, net::NNUENet)
    # king move: bucket may have changed; dirty forces refresh! with the now-restored board
    ptype(pieceon(b, from(m))) == KING && (acc.dirty = true; return)

    # bucket/flip unchanged for non-king moves; same values as when update! was called
    wbkt  = _king_bucket(acc.wksq);       wflip = _king_flip(acc.wksq)
    bbkt  = _king_bucket(acc.bksq ⊻ 56); bflip = _king_flip(acc.bksq)
    fw    = net.l0w
    mflag = Int((m >> 16) & 0xf)

    if mflag == 4
        _apply_ep!(acc, b, m, fw, wbkt, wflip, bbkt, bflip, Val(false))
    elseif promotion(m) ≠ PieceType(0)
        _apply_promo!(acc, b, m, fw, wbkt, wflip, bbkt, bflip, Val(false))
    else
        _apply_move!(acc, b, m, fw, wbkt, wflip, bbkt, bflip, Val(false))
    end
end

# Evaluate from accumulator. Lazy refresh on king moves. Returns centipawns (STM perspective).
function nnue_eval(acc::Accumulator, b::Board, net::NNUENet)::Int
    acc.dirty && refresh!(acc, b, net)
    us, them = sidetomove(b) == WHITE ? (acc.w, acc.b) : (acc.b, acc.w)
    bkt = _material_bucket(b)

    _crelu_pairwise_mul!(acc.stm_hidden, us)
    _crelu_pairwise_mul!(acc.ntm_hidden, them)
    copyto!(acc.hl1, 1,            acc.stm_hidden, 1, NNUE_HL2)
    copyto!(acc.hl1, NNUE_HL2 + 1, acc.ntm_hidden, 1, NNUE_HL2)

    _dense_forward!(acc.hl2, acc.hl1, net.l1w, net.l1b, bkt, true)
    _dense_forward!(acc.hl3, acc.hl2, net.l2w, net.l2b, bkt, true)
    _dense_forward!(acc.out, acc.hl3, net.l3w, net.l3b, bkt, false)

    return Int(round(acc.out[1] * NNUE_SCALE))
end

# Convenience: full recompute then evaluate (for root / testing)
function nnue_eval(b::Board, net::NNUENet)::Int
    acc = Accumulator()
    refresh!(acc, b, net)
    nnue_eval(acc, b, net)
end