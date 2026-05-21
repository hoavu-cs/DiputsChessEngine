# ============================================================
# Constant and shared variables
# ============================================================

const ∞        = 10_000_000
const MATE_SCORE =  9_000_000

const Δ = 256
const δ = 64
const Γ = 16384

abstract type NodeType end
struct PVNode  <: NodeType end
struct CutNode <: NodeType end
struct AllNode <: NodeType end

const _N_THREADS = max(1, Threads.nthreads() - 1)
const _NODE_COUNT   = fill(0, _N_THREADS)
const _SELDEPTH     = fill(0, _N_THREADS)
const _ROOT_DEPTH    = fill(0, _N_THREADS)

# ============================================================
# Transposition Table (shared)
# ============================================================

const _QS_PIECE_VAL = (100, 300, 300, 500, 900, 20000)

const TT_EXACT   = Int8(0)
const TT_LOWER   = Int8(1)
const TT_UPPER   = Int8(2)
const TT_EMPTY   = Int8(-1)
const TT_MAX_PLY = 256
const DEPTH_QS   = -1


struct TTEntry
    key::UInt64
    depth::Int32
    score::Int32
    flag::Int8
    is_pv::Bool
    best::UInt64
end

const TT_ENTRY_NULL = TTEntry(0, -1, 0, TT_EMPTY, false, Move(0))

tt::Vector{TTEntry}         = Vector{TTEntry}(undef, 1 << 22)
tt_mask::UInt64             = UInt64((1 << 22) - 1)
function init_tt()
    global tt      = Vector{TTEntry}(undef, 1 << 22)
    global tt_mask = UInt64((1 << 22) - 1)
    fill!(tt, TT_ENTRY_NULL)
end

function clear_tt()
    fill!(tt, TT_ENTRY_NULL)
end

function resize_tt(mb::Int)
    target = div(mb * 1024 * 1024, sizeof(TTEntry))
    size = 1 << 16
    while size * 2 ≤ target
        size *= 2
    end
    global tt      = Vector{TTEntry}(undef, size)
    global tt_mask = UInt64(size - 1)
    fill!(tt, TT_ENTRY_NULL)
end

@inline function probe_tt(key::UInt64, depth::Int, ply::Int)
    idx = Int(key & tt_mask) + 1
    entry = tt[idx]
    (entry.flag == TT_EMPTY || entry.key ≠ key) && return (false, 0, TT_EMPTY, false, Move(0), -1)
    score = entry.score
    if abs(score) > MATE_SCORE - TT_MAX_PLY
        score += score > 0 ? -ply : ply
    end
    return (entry.depth ≥ depth, score, entry.flag, entry.is_pv, entry.best, Int(entry.depth))
end

@inline function store_tt(key::UInt64, depth::Int, score::Int, flag::Int8, is_pv::Bool, best::UInt64, ply::Int)
    idx    = Int(key & tt_mask) + 1
    stored = score
    if abs(score) > MATE_SCORE - TT_MAX_PLY
        stored = score + (score > 0 ? ply : -ply)
    end
    tt[idx] = TTEntry(key, Int32(depth), Int32(stored), flag, is_pv, best)
end

init_tt()

# ============================================================
# Late Move Reductions
# ============================================================

const LMR_DEPTH_MAX = 99
const LMR_MOVES_MAX = 256
const LMR_TABLE = zeros(Int, LMR_DEPTH_MAX, LMR_MOVES_MAX)

function init_lmr_table!()
    for d in 1:LMR_DEPTH_MAX
        for i in 1:LMR_MOVES_MAX
            R = 1 + log(d) * log(i) / 3
            LMR_TABLE[d, i] = clamp(round(Int, R), 1, d - 1)
        end
    end
end

init_lmr_table!()

const _MAX_BUF_PLY = 512
# Pre-allocated per-thread, per-ply move buffers — eliminates ~2KB alloc per node
const _MOVE_BUFS = [[MoveList() for _ in 1:_MAX_BUF_PLY] for _ in 1:_N_THREADS]
# Separate buffers for singular searches to avoid clobbering the outer node's move list
const _SING_BUFS = [[MoveList() for _ in 1:_MAX_BUF_PLY] for _ in 1:_N_THREADS]

const _NO_PV   = UInt64[]
const _PV_BUFS = [[UInt64[] for _ in 1:256] for _ in 1:_N_THREADS]
const _CAND_PV = [UInt64[] for _ in 1:_N_THREADS]
const _ITER_PV = [UInt64[] for _ in 1:_N_THREADS]

# ============================================================
# NNUE (one accumulator per thread)
# ============================================================

const nnue_net  = load_nnue(joinpath(@__DIR__, "nnue.bin"))
const nnue_accs = [Accumulator() for _ in 1:_N_THREADS]

# ============================================================
# History Heuristics (per-thread, last dim = tid)
# ============================================================
"""
    Gravity moving average update:
    new = old + 0.35 * (bonus - old)
        ≈ 0.648 * old + 0.352 * bonus
    Then clamp to [-Γ, Γ].
"""

const history      = zeros(Int16, 2, 64, 64, _N_THREADS)
const cont_hist    = zeros(Int16, 64, 7, 64, 7, 2, _N_THREADS)
const cont_hist2   = zeros(Int16, 64, 7, 64, 7, 2, _N_THREADS)
const cap_hist     = zeros(Int16, 12, 64, 6, _N_THREADS)  
const eval_stack   = zeros(Int,   257, _N_THREADS)   
const killers      = fill(Move(0), 2, 256, _N_THREADS)
const move_stack   = fill((0, 0), 256, _N_THREADS)
const _MOVE_SCORES = zeros(Int,   256, 256, _N_THREADS)


function clear_history()
    fill!(history, 0)
    fill!(cont_hist, Int16(0))
    fill!(cont_hist2, Int16(0))
    fill!(cap_hist, Int16(0))
    fill!(pawn_hist, Int16(0))
    fill!(killers, Move(0))
    fill!(move_stack, (0, 0))
    fill!(_CORR_TABLE, Int16(0))
    fill!(_MINOR_TABLE, Int16(0))
    fill!(_MAJORW_TABLE, Int16(0))
    fill!(_MAJORB_TABLE, Int16(0))
end

@inline function update_cap_hist!(pc::Int, to_sq::Int, cap_pt::Int, bonus::Int, tid::Int)
    old = Int(cap_hist[pc, to_sq, cap_pt, tid])
    cap_hist[pc, to_sq, cap_pt, tid] = Int16(clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ))
end

@inline function update_history!(color::Int, from_sq::Int, to_sq::Int, bonus::Int, tid::Int)
    old = Int(history[color, from_sq, to_sq, tid])
    newv = clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ)
    @inbounds history[color, from_sq, to_sq, tid] = Int16(newv)
end

@inline function update_cont_hist!(
    stm::Int,
    ply::Int,
    cur_pt::Int, cur_to::Int,
    bonus::Int,
    tid::Int,
)
    prev_pt, prev_to   = ply ≥ 2 ? move_stack[ply - 1, tid] : (0, 0)
    prev2_pt, prev2_to = ply ≥ 3 ? move_stack[ply - 2, tid] : (0, 0)
    prev_pt == 0 && return
    @inbounds begin
        old = Int(cont_hist[cur_to, cur_pt, prev_to, prev_pt, stm, tid])
        cont_hist[cur_to, cur_pt, prev_to, prev_pt, stm, tid] = Int16(clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ))
    end
    if prev2_pt > 0
        @inbounds begin
            old2 = Int(cont_hist2[cur_to, cur_pt, prev2_to, prev2_pt, stm, tid])
            cont_hist2[cur_to, cur_pt, prev2_to, prev2_pt, stm, tid] = Int16(clamp(old2 + (bonus - old2) * δ ÷ Δ, -Γ, Γ))
        end
    end
end

const search_deadline      = Ref{UInt64}(typemax(UInt64))
const search_soft_deadline = Ref{UInt64}(typemax(UInt64))

# ============================================================
# Correction History
# ============================================================

# Pawn correction history: adjusts eval based on previous search outcomes
const _PAWN_HIST_SIZE = 1 << 14
const _PAWN_HIST_MASK = _PAWN_HIST_SIZE - 1
const pawn_hist       = zeros(Int16, 6, 64, _PAWN_HIST_SIZE, _N_THREADS)  # [pt, to, ph, tid]

const _CORR_SIZE    = 1 << 16
const _CORR_MASK    = _CORR_SIZE - 1
const _CORR_TABLE   = zeros(Int16, 2, _CORR_SIZE)  # [white, black]

@inline function pawn_hist_idx(b::Board)::Int
    pk = b.bb[BB_WP] ⊻ b.bb[BB_BP]
    pk = pk ⊻ (pk >> 33)
    pk *= 0xFF51AFD7ED558CCD
    pk = pk ⊻ (pk >> 33)
    Int(pk & _PAWN_HIST_MASK) + 1
end

@inline function pawn_hist_score(b::Board, pt::Int, to::Int, tid::Int)::Int
    Int(pawn_hist[pt, to, pawn_hist_idx(b), tid])
end

@inline function update_pawn_hist!(b::Board, pt::Int, to::Int, bonus::Int, tid::Int)
    idx = pawn_hist_idx(b)
    old = Int(pawn_hist[pt, to, idx, tid])
    pawn_hist[pt, to, idx, tid] = Int16(clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ))
end

@inline function corr_value(key::UInt64, color::Int)::Int
    Int(_CORR_TABLE[color, Int(key & _CORR_MASK) + 1])
end

@inline function corr_update!(key::UInt64, color::Int, bonus::Int)
    idx  = Int(key & _CORR_MASK) + 1
    old  = Int(_CORR_TABLE[color, idx])
    newv = clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ)
    _CORR_TABLE[color, idx] = Int16(newv)
end

# Minor piece correction history — same pattern, indexed by knight+bishop hash
const _MINOR_SIZE  = 1 << 14
const _MINOR_MASK  = _MINOR_SIZE - 1
const _MINOR_TABLE = zeros(Int16, 2, _MINOR_SIZE)

@inline function minor_key(b::Board)::UInt64
    (b.bb[BB_WN] | b.bb[BB_BN]) ⊻ (b.bb[BB_WB] | b.bb[BB_BB])
end

@inline function minor_corr_value(key::UInt64, color::Int)::Int
    Int(_MINOR_TABLE[color, Int(key & _MINOR_MASK) + 1])
end

@inline function update_minor_corr!(key::UInt64, color::Int, bonus::Int)
    idx  = Int(key & _MINOR_MASK) + 1
    old  = Int(_MINOR_TABLE[color, idx])
    newv = clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ)
    _MINOR_TABLE[color, idx] = Int16(newv)
end

# Major piece correction history — keys on major material (queens + rooks), split by WHITE/BLACK.
const _MAJORW_SIZE = 1 << 14
const _MAJORW_MASK = _MAJORW_SIZE - 1
const _MAJORW_TABLE = zeros(Int16, 2, _MAJORW_SIZE)
const _MAJORB_SIZE = 1 << 14
const _MAJORB_MASK = _MAJORB_SIZE - 1
const _MAJORB_TABLE = zeros(Int16, 2, _MAJORB_SIZE)

@inline function major_corr_value_w(key::UInt64, color::Int)::Int
    Int(_MAJORW_TABLE[color, Int(key & _MAJORW_MASK) + 1])
end

@inline function major_corr_value_b(key::UInt64, color::Int)::Int
    Int(_MAJORB_TABLE[color, Int(key & _MAJORB_MASK) + 1])
end

@inline function update_major_corr_w!(key::UInt64, color::Int, bonus::Int)
    idx  = Int(key & _MAJORW_MASK) + 1
    old  = Int(_MAJORW_TABLE[color, idx])
    newv = clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ)
    _MAJORW_TABLE[color, idx] = Int16(newv)
end

@inline function update_major_corr_b!(key::UInt64, color::Int, bonus::Int)
    idx  = Int(key & _MAJORB_MASK) + 1
    old  = Int(_MAJORB_TABLE[color, idx])
    newv = clamp(old + (bonus - old) * δ ÷ Δ, -Γ, Γ)
    _MAJORB_TABLE[color, idx] = Int16(newv)
end

# ============================================================
# Move Ordering
# ============================================================

const _SCORE_HASH        = 1_000_000
const _SCORE_PROMO       =   900_000
const _SCORE_CAPTURE     =   100_000   # good captures (SEE ≥ 0): above killers
const _SCORE_KILLER      =    90_000
                                        # quiets: history score ∈ [-16384, 16384]
const _SCORE_BAD_CAPTURE =  -100_000   # bad captures (SEE < 0): below all quiets

@inline function score_move(
    b::Board,
    m::UInt64,
    tt_move::UInt64,
    k1::UInt64,
    k2::UInt64,
    ply::Int;
    tid::Int = 1,
)::Int
    m == tt_move && return _SCORE_HASH

    promo = promotion(m)
    if promo ≠ PieceType(0)
        return _SCORE_PROMO + promo.val
    end

    if moveiscapture(b, m)
        see_val = see(b, m)
        pc      = Int(b.pieces[from(m).val])            # moving piece 1..12
        cap_pt  = max(1, ptype(pieceon(b, to(m))).val)   # 0 on en passant → treat as pawn
        ch      = @inbounds Int(cap_hist[pc, to(m).val, cap_pt, tid]) * _SEE_VAL[QUEEN] ÷ Γ
        return see_val ≥ 0 ? _SCORE_CAPTURE + see_val + ch : _SCORE_BAD_CAPTURE + see_val + ch
    end

    m == k1 && return _SCORE_KILLER
    m == k2 && return _SCORE_KILLER - 1

    color              = sidetomove(b) == WHITE ? 1 : 2
    cur_pt             = ptype(pieceon(b, from(m))).val
    prev_pt,  prev_to  = ply ≥ 2 ? move_stack[ply - 1, tid] : (0, 0)
    prev2_pt, prev2_to = ply ≥ 3 ? move_stack[ply - 2, tid] : (0, 0)
    ch  = prev_pt  > 0 ? @inbounds(Int(cont_hist[to(m).val,  cur_pt, prev_to,  prev_pt,  color, tid])) : 0
    ch2 = prev2_pt > 0 ? @inbounds(Int(cont_hist2[to(m).val, cur_pt, prev2_to, prev2_pt, color, tid])) : 0
    ph  = pawn_hist_score(b, cur_pt, to(m).val, tid)
    return @inbounds(history[color, from(m).val, to(m).val, tid]) + (ch ÷ 2) + (ch2 ÷ 3) + (ph ÷ 2)
end

function sort_moves!(
    b::Board,
    ml::AbstractVector{Move},
    ply::Int,
    tt_best::UInt64,
    k1::UInt64,
    k2::UInt64,
    tid::Int,
)
    n      = length(ml)
    scores = @view _MOVE_SCORES[1:n, ply, tid]

    for i in 1:n
        scores[i] = score_move(b, ml[i], tt_best, k1, k2, ply; tid=tid)
    end

    for i in 2:n
        tmp_m = ml[i]
        tmp_s = scores[i]
        j = i - 1
        while j ≥ 1 && scores[j] < tmp_s
            ml[j+1]     = ml[j]
            scores[j+1] = scores[j]
            j -= 1
        end
        ml[j+1]     = tmp_m
        scores[j+1] = tmp_s
    end
end

# ============================================================
# Quiescence Search
# ============================================================

function quiescence(b::Board, α::Int, β::Int, ply::Int, key_history::Vector{UInt64}, tid::Int)::Int
    _NODE_COUNT[tid] += 1
    _SELDEPTH[tid] = max(_SELDEPTH[tid], ply)
    search_stopped[] && return 0

    # Probe TT with depth -1 
    hit, tt_score, tt_flag, _, _, _ = probe_tt(b.key, DEPTH_QS, ply)
    if hit
        if (tt_flag == TT_LOWER && tt_score ≥ β) || (tt_flag == TT_UPPER && tt_score ≤ α)
            return tt_score
        end
    end

    # Exhaust captures
    stand_pat = nnue_eval(nnue_accs[tid], b, nnue_net)
    if stand_pat ≥ β
        !search_stopped[] && store_tt(b.key, DEPTH_QS, stand_pat, TT_LOWER, false, Move(0), ply)
        return stand_pat
    end
    α = max(α, stand_pat)

    cap_buf = _MOVE_BUFS[tid][min(ply, _MAX_BUF_PLY - 1) + 1]
    generate_moves!(cap_buf, b)
    j = 0
    for i in 1:cap_buf.count
        m = cap_buf.moves[i]
        if moveiscapture(b, m)
            j += 1
            cap_buf.moves[j] = m
        end
    end
    if j == 0
        !search_stopped[] && store_tt(b.key, DEPTH_QS, α, TT_UPPER, false, Move(0), ply)
        return α
    end
    cap_view = @view cap_buf.moves[1:j]

    sort_moves!(b, cap_view, ply, Move(0), Move(0), Move(0), tid)

    best      = stand_pat
    best_move = Move(0)
    for m in cap_view
        search_stopped[] && break

        # Skip captures that lose material — they can't raise alpha from stand_pat.
        see(b, m) < 0 && continue

        update!(nnue_accs[tid], b, m, nnue_net)
        u  = domove!(b, m)
        if was_illegal(b)
            undomove!(b, u)
            undo_update!(nnue_accs[tid], b, m, nnue_net)
            continue
        end
        push!(key_history, b.key)
        sc = -quiescence(b, -β, -α, ply + 1, key_history, tid)
        pop!(key_history)
        undomove!(b, u)
        undo_update!(nnue_accs[tid], b, m, nnue_net)

        if sc > best
            best      = sc
            best_move = m
        end
        α = max(α, sc)
        α ≥ β && break
    end

    # Store result in TT at depth -1
    if !search_stopped[]
        flag = best ≥ β ? TT_LOWER : TT_UPPER
        store_tt(b.key, DEPTH_QS, best, flag, false, best_move, ply)
    end

    return best
end

# ============================================================
# Negamax
# ============================================================

function negamax(
    ::Type{NT},
    b::Board,
    depth::Int,
    α::Int,
    β::Int,
    ply::Int,
    key_history::Vector{UInt64},
    tid::Int,
    pv::Vector{UInt64} = _NO_PV;
    excluded_move::UInt64 = Move(0),
)::Int where {NT <: NodeType}

    is_singular = excluded_move ≠ Move(0)
    _NODE_COUNT[tid] += 1
    _SELDEPTH[tid] = max(_SELDEPTH[tid], ply)
    search_stopped[] && return 0
    ply > 230 && return quiescence(b, α, β, ply, key_history, tid)
    if _NODE_COUNT[tid] & 0x3FFF == 0 && time_ns() ≥ search_deadline[]
        search_stopped[] = true
        return 0
    end

    if ply > 1
        cnt = 0
        for k in key_history
            k == b.key && (cnt += 1)
            cnt ≥ 2 && return 0
        end
        isdraw(b) && return 0
    end

    in_check = ischeck(b)
    stm      = sidetomove(b) == WHITE ? 1 : 2

    if depth == 0
        if !in_check
            return quiescence(b, α, β, ply, key_history, tid)
        end
        depth = 1
    end

    # TT look up for alpha-beta pruning
    hit, tt_score, tt_flag, tt_is_pv, tt_best, tt_stored_depth = probe_tt(b.key, depth, ply)
    is_pv_node  = (NT === PVNode)
    is_cut_node = (NT === CutNode)
    if hit && !is_singular && ply > 1
        if tt_flag == TT_EXACT
            return tt_score
        elseif tt_flag == TT_LOWER
            α = max(α, tt_score)
        elseif tt_flag == TT_UPPER
            β = min(β, tt_score)
        end
        α ≥ β && return tt_score
    end

    # Internal iterative reduction (skip at AllNodes)
    depth -= (tt_best == Move(0) && depth ≥ 3 && (is_pv_node || is_cut_node)) && (ply > 1) ? 1 : 0

    # Raw NNUE eval
    raw_eval = nnue_eval(nnue_accs[tid], b, nnue_net)

    # Apply pawn, minor & non-pawn correction history to raw eval
    eval = raw_eval + (corr_value(b.bb[BB_WP] | b.bb[BB_BP], stm)
                     + minor_corr_value(minor_key(b), stm)
                     + major_corr_value_w(b.bb[BB_WQ] | b.bb[BB_WR], stm) ÷ 2
                     + major_corr_value_b(b.bb[BB_BQ] | b.bb[BB_BR], stm) ÷ 2) ÷ Δ

    # TT score overrides if it provides a tighter bound
    if tt_flag == TT_EXACT ||
       (tt_flag == TT_LOWER && tt_score > eval) ||
       (tt_flag == TT_UPPER && tt_score < eval)
        eval = tt_score
    end
    
    eval = clamp(eval, -(MATE_SCORE - TT_MAX_PLY), MATE_SCORE - TT_MAX_PLY)
    eval_stack[ply, tid] = eval
    improving = !in_check && ply ≥ 3 && eval > eval_stack[ply - 2, tid]

    # Mini-probcut: TT lower bound well above beta ⟹ prune
    if ply > 1 && (tt_flag == TT_LOWER || tt_flag == TT_EXACT) && tt_stored_depth ≥ depth - 3 &&
       tt_score ≥ β + 500 && abs(β) < MATE_SCORE - TT_MAX_PLY &&
       abs(tt_score) < MATE_SCORE - TT_MAX_PLY && !is_singular
        return β + 500
    end

    # Reverse futility pruning (multiplier interpolated 120→175 over depths 1–7)
    if depth ≤ 7 && !in_check && !is_pv_node && !is_singular
        rfp_mult = (140, 145, 155, 160, 165, 175, 200)[depth]
        if eval ≥ β + rfp_mult * depth - 85 * improving
            if tt_best == Move(0) || history[stm, from(tt_best).val, to(tt_best).val, tid] > 7000
                return (eval + β) ÷ 2
            end
        end
    end

    # Razoring: at depth 1, if eval is far below alpha, just return qsearch
    if depth == 1 && !in_check && !is_pv_node && eval + 300 ≤ α
        return quiescence(b, α, β, ply, key_history, tid)
    end

    # Null move pruning
    if depth ≥ 3 && !in_check && eval ≥ β && !is_pv_node && !is_singular
        has_piece = false
        side = sidetomove(b)
        for pt in (QUEEN, ROOK, BISHOP, KNIGHT)
            for _ in pieces(b, side, pt)
                has_piece = true
                break
            end
            has_piece && break
        end
        if has_piece
            R = min(4 + div(depth, 3) + min(div(eval - β, 200), 3), depth - 1)
            u = donullmove!(b)
            move_stack[ply, tid] = (0, 0)
            sc = -negamax(CutNode, b, depth - 1 - R, -β, -β + 1, ply + 1, key_history, tid)
            undomove!(b, u)
            sc ≥ β && return sc
        end
    end

    buf_idx  = min(ply, _MAX_BUF_PLY - 1) + 1
    ml_buf   = is_singular ? _SING_BUFS[tid][buf_idx] : _MOVE_BUFS[tid][buf_idx]
    generate_moves!(ml_buf, b)
    ml       = view(ml_buf.moves, 1:ml_buf.count)

    k1  = ply > 1 ? killers[1, ply, tid] : Move(0)
    k2  = ply > 1 ? killers[2, ply, tid] : Move(0)
    sort_moves!(b, ml, ply, tt_best, k1, k2, tid)

    best_score      = -∞
    best_move       = Move(0)
    flag            = TT_UPPER
    searched_quiets   = Move[]
    searched_captures = Move[]
    legal_moves       = 0
    α_0 = α

    for (i, m) in enumerate(ml)
        m == excluded_move && continue

        lmr        = legal_moves > 1 && depth ≥ 3 && promotion(m) == PieceType(0)
        is_capture = moveiscapture(b, m)
        is_quiet   = !is_capture && promotion(m) == PieceType(0)
        cur_pt     = ptype(pieceon(b, from(m))).val

        # Late move pruning
        if depth ≤ 8 && !in_check && is_quiet && !is_pv_node
            length(searched_quiets) ≥ (5 + depth * depth) ÷ (2 - Int(improving)) && continue
        end

        # SEE pruning + cache for LMR tweak below.
        see_val = (!in_check && !is_pv_node && !is_singular) ? see(b, m) : 0
        if depth ≤ 8 && !in_check && !is_pv_node && !is_singular
            see_threshold = is_capture ? -40 * depth : -80 * depth * depth
            see_val < see_threshold && continue
        end

        # Singular extension: if the TT move is clearly better than all others,
        # extend it by 1. Prevents recursive singular searches via excluded_move guard.
        ext = 0
        if m == tt_best && !is_singular && ply > 1 && depth ≥ 6 && tt_flag ≠ TT_UPPER && ply ≤ 2 * _ROOT_DEPTH[tid] &&
            abs(tt_score) < MATE_SCORE - TT_MAX_PLY && tt_stored_depth ≥ depth - 3

            singular_β  = tt_score - 6 * depth 
            singular_depth = (depth ÷ 2) + 1
            sing_score = negamax(CutNode, b, singular_depth, singular_β - 1, singular_β,
                                 ply, key_history, tid; excluded_move = m)
            if sing_score < singular_β
                ext = 1
                if sing_score < singular_β - 20 
                    ext += 1 # double extension
                end
                if sing_score < singular_β - 40 
                    ext += 1 # triple extension
                end
            elseif singular_β ≥ β
                return singular_β  # multicut
            end
        end
        new_depth = depth - 1 + ext

        update!(nnue_accs[tid], b, m, nnue_net)
        u  = domove!(b, m)
        if was_illegal(b)
            undomove!(b, u)
            undo_update!(nnue_accs[tid], b, m, nnue_net)
            continue
        end
        legal_moves += 1
        push!(key_history, b.key)
        move_stack[ply, tid] = (cur_pt, to(m).val)

        killers[1, ply + 1, tid] = Move(0)
        killers[2, ply + 1, tid] = Move(0)

        if lmr
            R = LMR_TABLE[depth, min(legal_moves, LMR_MOVES_MAX)]
            ischeck(b)      && (R = max(1, R - 1))
            is_capture      && (R = max(1, R - 1))
            NT === AllNode  && (R += 1)
            NT === CutNode  && (R += 1)
            if !is_pv_node && ply ≥ 5 &&
                eval < eval_stack[ply - 2, tid] < eval_stack[ply - 4, tid]
                R += 1
            end
            R = min(R, depth - 1)
        else
            R = 0
        end

        # PV: first legal ⟹ PVNode; others ⟹ CutNode
        # Cut: first legal ⟹ AllNode; others ⟹ CutNode
        # All: all children ⟹ CutNode
        child_pv  = _NO_PV
        child_idx = min(ply + 1, 256)
        if legal_moves == 1 && is_pv_node
            child_pv = _PV_BUFS[tid][child_idx]; empty!(child_pv)
            sc = -negamax(PVNode, b, new_depth, -β, -α, ply + 1, key_history, tid, child_pv)
        elseif is_cut_node && legal_moves == 1
            sc = -negamax(AllNode, b, new_depth - R, -α - 1, -α, ply + 1, key_history, tid)
            if sc > α && R > 0
                sc = -negamax(AllNode, b, new_depth, -α - 1, -α, ply + 1, key_history, tid)
            end
        else
            sc = -negamax(CutNode, b, new_depth - R, -α - 1, -α, ply + 1, key_history, tid)
            if sc > α
                R > 0 && (sc = -negamax(CutNode, b, new_depth, -α - 1, -α, ply + 1, key_history, tid))
                if is_pv_node && sc > α && sc < β
                    child_pv = _PV_BUFS[tid][child_idx]; empty!(child_pv)
                    sc = -negamax(PVNode, b, new_depth, -β, -α, ply + 1, key_history, tid, child_pv)
                end
            end
        end

        pop!(key_history)
        undomove!(b, u)
        undo_update!(nnue_accs[tid], b, m, nnue_net)

        if sc > best_score
            best_score = sc
            best_move  = m

            if sc > α
                α    = sc
                flag = TT_EXACT
                if is_pv_node
                    empty!(pv)
                    push!(pv, m)
                    append!(pv, child_pv)
                end

                if α ≥ β
                    flag = TT_LOWER
                    bonus = depth * depth

                    if is_quiet
                        killers[2, ply, tid] = killers[1, ply, tid]
                        killers[1, ply, tid] = m
                        update_history!(stm, from(m).val, to(m).val, bonus, tid)
                        update_cont_hist!(stm, ply, cur_pt, to(m).val, bonus, tid)
                        update_pawn_hist!(b, cur_pt, to(m).val, bonus, tid)
                        for qm in searched_quiets
                            qm_pt = ptype(pieceon(b, from(qm))).val
                            update_history!(stm, from(qm).val, to(qm).val, -bonus, tid)
                            update_cont_hist!(stm, ply, qm_pt, to(qm).val, -bonus, tid)
                            update_pawn_hist!(b, qm_pt, to(qm).val, -bonus, tid)
                        end
                    elseif is_capture && depth > 2
                        pc     = Int(b.pieces[from(m).val])
                        cap_pt = max(1, ptype(pieceon(b, to(m))).val)
                        update_cap_hist!(pc, to(m).val, cap_pt, bonus, tid)
                        for cm in searched_captures
                            cm_pc     = Int(b.pieces[from(cm).val])
                            cm_cap_pt = max(1, ptype(pieceon(b, to(cm))).val)
                            update_cap_hist!(cm_pc, to(cm).val, cm_cap_pt, -bonus, tid)
                        end
                    end
                    break
                end
            end
        end

        is_quiet   && push!(searched_quiets, m)
        is_capture && push!(searched_captures, m)
    end

    if legal_moves == 0 && !search_stopped[]
        return in_check ? -(MATE_SCORE - ply) : 0
    end

    if !search_stopped[] && !in_check && depth ≥ 2
        diff = best_score - raw_eval
        best_is_capture = best_move ≠ Move(0) && moveiscapture(b, best_move)
        direction_ok = (best_score > α_0) ? (diff > 0) : (diff < 0)          # fail-high: up only; fail-low: down only
        if !best_is_capture && direction_ok
            bonus = clamp(diff * depth * Δ, -Γ, Γ)
            corr_update!(b.bb[BB_WP] | b.bb[BB_BP], stm, bonus)
            update_minor_corr!(minor_key(b), stm, bonus)
            update_major_corr_w!(b.bb[BB_WQ] | b.bb[BB_WR], stm, bonus)
            update_major_corr_b!(b.bb[BB_BQ] | b.bb[BB_BR], stm, bonus)
        end
    end

    !search_stopped[] && !is_singular && store_tt(b.key, depth, best_score, flag, is_pv_node, best_move, ply)
    return best_score
end

# ============================================================
# Iterative Deepening Root
# ============================================================

function search(b::Board, max_depth::Int, tid::Int; depth_offset::Int=0)::UInt64
    fill!(@view(killers[:, :, tid]), Move(0))
    refresh!(nnue_accs[tid], b, nnue_net)

    start_ns         = time_ns()
    best_move        = Move(0)
    _NODE_COUNT[tid] = 0
    _SELDEPTH[tid]   = 0
    _ROOT_DEPTH[tid] = 0
    key_history      = copy(game_key_history)

    prev_score = 0

    for depth in (1 + depth_offset):max_depth
        _ROOT_DEPTH[tid] = depth
        search_stopped[] && break

        prev_best    = best_move
        had_asp_fail = false
        window       = 35
        asp_α        = depth ≥ 6 ? prev_score - window : -∞
        asp_β        = depth ≥ 6 ? prev_score + window :  ∞
        best_score   = prev_score
        iter_pv      = _ITER_PV[tid]

        while true
            search_stopped[] && break
            empty!(iter_pv)
            sc = negamax(PVNode, b, depth, asp_α, asp_β, 1, key_history, tid, iter_pv)
            search_stopped[] && break

            if sc ≤ asp_α
                had_asp_fail = true
                window *= 2
                asp_α = window ≥ 200 ? -∞ : max(asp_α - window, -∞)
            elseif sc ≥ asp_β
                had_asp_fail = true
                !isempty(iter_pv) && (best_move = iter_pv[1]; best_score = sc)
                window *= 2
                asp_β = window ≥ 200 ?  ∞ : min(asp_β + window, ∞)
            else
                best_score = sc
                !isempty(iter_pv) && (best_move = iter_pv[1])
                break
            end
        end

        if !search_stopped[] && best_move ≠ Move(0)
            prev_score = best_score
            if tid == 1
                store_tt(b.key, depth, best_score, TT_EXACT, true, best_move, 1)
                elapsed_ms = div(time_ns() - start_ns, 1_000_000)
                node_count = sum(_NODE_COUNT)
                seldepth   = maximum(_SELDEPTH)
                nps        = elapsed_ms > 0 ? div(node_count * 1000, elapsed_ms) : 0
                pv_str     = join(tostring.(iter_pv), " ")
                println("info depth $depth seldepth $seldepth score cp $best_score time $elapsed_ms nodes $node_count nps $nps pv $pv_str")
                flush(stdout)
            end
        end

        now = time_ns()
        now ≥ search_deadline[] && break
        if now ≥ search_soft_deadline[]
            !had_asp_fail && best_move ≠ Move(0) && best_move == prev_best && break
        end
    end

    return best_move
end

# ============================================================
# SMP
# ============================================================

function smp_search(b::Board, max_depth::Int, time_soft::Int, time_hard::Int)::UInt64
    start_ns = time_ns()
    inf_time = time_hard ≥ typemax(Int) >> 20
    search_deadline[]      = inf_time ? start_ns + 30_000_000_000 : start_ns + UInt64(time_hard) * 1_000_000
    search_soft_deadline[] = inf_time ? typemax(UInt64)           : start_ns + UInt64(time_soft) * 1_000_000

    n = _N_THREADS
    if n == 1
        return search(b, max_depth, 1)
    end

    # reset node counts for all threads
    for tid in 1:n
        _NODE_COUNT[tid] = 0
    end

    # threads 2 to n will just populate the TT.
    # Alternate depth offset so helpers are out of phase with the main thread (doing iterative deepening at some offset depth)
    tasks = Vector{Task}(undef, n)
    for tid in 2:n
        b_copy = deepcopy(b)
        offset = (tid - 1) % 2  # tid=2 → 1, tid=3 → 0, tid=4 → 1, ...
        tasks[tid] = Threads.@spawn search(b_copy, max_depth, tid; depth_offset=offset)
    end

    # thread 1 is the master thread that reports the result. 
    result = search(b, max_depth, 1)
    search_stopped[] = true

    for tid in 2:n
        wait(tasks[tid])
    end

    return result
end