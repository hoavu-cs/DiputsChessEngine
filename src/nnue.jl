const NNUE_INPUT   = 768
const NNUE_HL      = 1024
const NNUE_BUCKETS = 8
const NNUE_SCALE   = 400
const NNUE_QA      = 255
const NNUE_QB      = 64
const NNUE_IB           = 10   # king input buckets (ChessBucketsMirrored)
const NNUE_FINNY_SLOTS  = 2 * NNUE_IB  # mirror (0/1) × bucket (0..9) = 20 slots per perspective

# King bucket layout: 32 entries indexed by rank*4 + mirror_file
# rank=0(rank1)..7(rank8), mirror_file = min(file, 7-file)
const _IB_LAYOUT = (
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
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
    fw::Matrix{Int16}   # NNUE_HL × (NNUE_INPUT × NNUE_IB)  
    fb::Vector{Int16}   # NNUE_HL
    ow::Matrix{Int16}   # 2*NNUE_HL × NNUE_BUCKETS
    ob::Vector{Int16}   # NNUE_BUCKETS
end

# quantised.bin layout (from train_ib.rs):
#   l0w : Int16[1024, 768 × NNUE_IB]   (QA=255, factoriser merged)
#   l0b : Int16[1024]
#   l1w : Int16[2*1024, NNUE_BUCKETS]  (QB=64, transposed)
#   l1b : Int16[NNUE_BUCKETS]
function load_nnue(path::String)::NNUENet
    open(path, "r") do io
        fw = Matrix{Int16}(undef, NNUE_HL, NNUE_INPUT * NNUE_IB)
        read!(io, fw)
        fb = Vector{Int16}(undef, NNUE_HL)
        read!(io, fb)
        ow = Matrix{Int16}(undef, 2 * NNUE_HL, NNUE_BUCKETS)
        read!(io, ow)
        ob = Vector{Int16}(undef, NNUE_BUCKETS)
        read!(io, ob)
        NNUENet(fw, fb, ow, ob)
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

# Finny slot index (1-based, range 1..NNUE_FINNY_SLOTS=20)
# White perspective: bucket from physical wksq, mirror from physical wksq file
@inline _finny_slot_w(wksq::Int) =
    ((wksq & 7) >= 4 ? NNUE_IB : 0) + _king_bucket(wksq) + 1

# Black perspective: bucket from rank-flipped bksq, mirror from physical bksq file
@inline _finny_slot_b(bksq::Int) =
    ((bksq & 7) >= 4 ? NNUE_IB : 0) + _king_bucket(bksq ⊻ 56) + 1

# Two NNUE_HL accumulators (white / black perspective) plus king state and Finny caches
mutable struct Accumulator
    w::Vector{Int32}
    b::Vector{Int32}
    wksq::Int    # white king std 0-based square
    bksq::Int    # black king std 0-based square
    dirty::Bool  # true ⟹ needs refresh! before eval
    finny_w::Vector{FinnyEntry}   # NNUE_FINNY_SLOTS entries for white perspective
    finny_b::Vector{FinnyEntry}   # NNUE_FINNY_SLOTS entries for black perspective
end

Accumulator() = Accumulator(
    zeros(Int32, NNUE_HL), zeros(Int32, NNUE_HL),
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

# Refresh both perspectives via the Finny cache. Called on king moves and at search root.
function refresh!(acc::Accumulator, b::Board, net::NNUENet)
    wksq  = Int(trailing_zeros(b.bb[BB_WK]))
    bksq  = Int(trailing_zeros(b.bb[BB_BK]))
    wbkt  = _king_bucket(wksq);       wflip = _king_flip(wksq)
    bbkt  = _king_bucket(bksq ⊻ 56); bflip = _king_flip(bksq)
    wslot = _finny_slot_w(wksq)
    bslot = _finny_slot_b(bksq)
    _refresh_finny!(acc.w, net.fw, net.fb, acc.finny_w[wslot], wbkt, wflip, b, false)
    _refresh_finny!(acc.b, net.fw, net.fb, acc.finny_b[bslot], bbkt, bflip, b, true)
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
    fw    = net.fw
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
    fw    = net.fw
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
    ow  = net.ow
    out = Int64(0)
    @inbounds @simd for i in 1:NNUE_HL
        uv = clamp(us[i],   Int32(0), Int32(NNUE_QA))
        tv = clamp(them[i], Int32(0), Int32(NNUE_QA))
        out += Int64(uv) * uv * Int64(ow[i,           bkt])
        out += Int64(tv) * tv * Int64(ow[NNUE_HL + i, bkt])
    end
    out = div(out, Int64(NNUE_QA)) + @inbounds Int64(net.ob[bkt])
    return Int(div(out * Int64(NNUE_SCALE), Int64(NNUE_QA * NNUE_QB)))
end

# Convenience: full recompute then evaluate (for root / testing)
function nnue_eval(b::Board, net::NNUENet)::Int
    acc = Accumulator()
    refresh!(acc, b, net)
    nnue_eval(acc, b, net)
end
