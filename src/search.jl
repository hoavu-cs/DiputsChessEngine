# search.jl — NNUE search (lazy SMP)

const ∞        = 10_000_000
const MATE_SCORE =  9_000_000

const MAX_HISTORY = 16384
const Δ = 256
const δ = 64
const Γ = 16384

abstract type NodeType end
struct PVNode    <: NodeType end
struct NonPVNode <: NodeType end

const _N_THREADS = max(1, Threads.nthreads())
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

tt::Vector{TTEntry}  = Vector{TTEntry}(undef, 1 << 22)
tt_mask::UInt64      = UInt64((1 << 22) - 1)

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
    idx   = Int(key & tt_mask) + 1
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

@inline function lmr_reduction(depth::Int, i::Int, in_check::Bool, is_capture::Bool, is_pv::Bool)::Int
    r = LMR_TABLE[depth, min(i, LMR_MOVES_MAX)]
    in_check   && (r = max(1, r - 1))
    is_capture && (r = max(1, r - 1))
    !is_pv      && (r += 1)
    return min(r, depth - 1)
end

# ============================================================
# NNUE (one accumulator per thread)
# ============================================================

const nnue_net  = load_nnue(joinpath(@__DIR__, "nnue_hl_1024.bin"))
const nnue_accs = [Accumulator() for _ in 1:_N_THREADS]

# ============================================================
# History Heuristics (per-thread, last dim = tid)
# ============================================================

const history      = zeros(Int,   2, 64, 64, _N_THREADS)
const cont_hist    = zeros(Int16, 64, 7, 64, 7, 2, _N_THREADS)
const cont_hist2   = zeros(Int16, 64, 7, 64, 7, 2, _N_THREADS)
const eval_stack   = zeros(Int,   257, _N_THREADS)   # [ply+1, tid]; [1, tid] = root eval
const killers      = fill(Move(0), 2, 256, _N_THREADS)
const move_stack   = fill((0, 0), 256, _N_THREADS)
const _MOVE_SCORES = zeros(Int,   256, 256, _N_THREADS)

function clear_history()
    fill!(history, 0)
    fill!(cont_hist, Int16(0))
    fill!(cont_hist2, Int16(0))
    fill!(killers, Move(0))
    fill!(move_stack, (0, 0))
    fill!(_CORR_TABLE, Int16(0))
    fill!(_MINOR_TABLE, Int16(0))
    fill!(_MAJORW_TABLE, Int16(0))
    fill!(_MAJORB_TABLE, Int16(0))
end

@inline function update_history!(color::Int, from_sq::Int, to_sq::Int, bonus::Int, tid::Int)
    clamped = clamp(bonus, -MAX_HISTORY, MAX_HISTORY)
    @inbounds history[color, from_sq, to_sq, tid] +=
        clamped - history[color, from_sq, to_sq, tid] * abs(clamped) ÷ MAX_HISTORY
end

@inline function update_cont_hist!(
    stm::Int,
    ply::Int,
    cur_pt::Int, cur_to::Int,
    bonus::Int,
    tid::Int,
)
    prev_pt, prev_to   = ply > 1 ? move_stack[ply, tid]     : (0, 0)
    prev2_pt, prev2_to = ply > 2 ? move_stack[ply - 1, tid] : (0, 0)
    prev_pt == 0 && return
    clamped = clamp(bonus, -MAX_HISTORY, MAX_HISTORY)
    @inbounds begin
        old = Int(cont_hist[cur_to, cur_pt, prev_to, prev_pt, stm, tid])
        cont_hist[cur_to, cur_pt, prev_to, prev_pt, stm, tid] = Int16(old + clamped - old * abs(clamped) ÷ MAX_HISTORY)
    end
    if prev2_pt > 0
        clamped2 = clamp(bonus, -MAX_HISTORY, MAX_HISTORY)
        @inbounds begin
            old2 = Int(cont_hist2[cur_to, cur_pt, prev2_to, prev2_pt, stm, tid])
            cont_hist2[cur_to, cur_pt, prev2_to, prev2_pt, stm, tid] = Int16(old2 + clamped2 - old2 * abs(clamped2) ÷ MAX_HISTORY)
        end
    end
end

const search_deadline = Ref{UInt64}(typemax(UInt64))

# ============================================================
# Correction History
# ============================================================

# Pawn correction history: adjusts eval based on previous search outcomes
const _CORR_SIZE    = 1 << 16
const _CORR_MASK    = _CORR_SIZE - 1
const _CORR_TABLE   = zeros(Int16, 2, _CORR_SIZE)  # [white, black]
# Correction history Δ, δ, Γ now live in Δ, δ, Γ

@inline function corr_value(key::UInt64, color::Int)::Int
    Int(_CORR_TABLE[color, Int(key & _CORR_MASK) + 1])
end

"""
    Gravity moving average update:
    new = old + 0.25 * (bonus - old)
        = 0.75 * old + 0.25 * bonus
    Then clamp to [-Γ, Γ].
"""
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

const _SCORE_HASH    = 1_000_000
const _SCORE_PROMO   =   900_000
const _SCORE_CAPTURE =   100_000
const _SCORE_KILLER  =    90_000

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
        vt       = ptype(pieceon(b, to(m))).val
        victim   = clamp(vt, 1, 6)
        attacker = ptype(pieceon(b, from(m))).val
        return _SCORE_CAPTURE + _QS_PIECE_VAL[victim] * 10 - attacker
    end

    m == k1 && return _SCORE_KILLER
    m == k2 && return _SCORE_KILLER - 1

    color              = sidetomove(b) == WHITE ? 1 : 2
    cur_pt             = ptype(pieceon(b, from(m))).val
    prev_pt,  prev_to  = ply > 1 ? move_stack[ply, tid]     : (0, 0)
    prev2_pt, prev2_to = ply > 2 ? move_stack[ply - 1, tid] : (0, 0)
    ch  = prev_pt  > 0 ? @inbounds(Int(cont_hist[to(m).val,  cur_pt, prev_to,  prev_pt,  color, tid])) : 0
    ch2 = prev2_pt > 0 ? @inbounds(Int(cont_hist2[to(m).val, cur_pt, prev2_to, prev2_pt, color, tid])) : 0
    return @inbounds(history[color, from(m).val, to(m).val, tid]) + (ch ÷ 2) + (ch2 ÷ 3)
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

    hit, tt_score, tt_flag, _, _, _ = probe_tt(b.key, DEPTH_QS, ply)
    if hit
        if (tt_flag == TT_LOWER && tt_score ≥ β) || (tt_flag == TT_UPPER && tt_score ≤ α)
            return tt_score
        end
    end

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
    tid::Int;
    excluded_move::UInt64 = Move(0),
)::Int where {NT <: NodeType}

    is_singular = excluded_move ≠ Move(0)
    _NODE_COUNT[tid] += 1
    _SELDEPTH[tid] = max(_SELDEPTH[tid], ply)
    search_stopped[] && return 0
    if time_ns() ≥ search_deadline[]
        search_stopped[] = true
        return 0
    end

    cnt = 0
    for k in key_history
        k == b.key && (cnt += 1)
        cnt ≥ 2 && return 0
    end
    isdraw(b) && return 0

    buf_idx  = min(ply, _MAX_BUF_PLY - 1) + 1
    ml_buf   = _MOVE_BUFS[tid][buf_idx]
    generate_moves!(ml_buf, b)
    ml       = view(ml_buf.moves, 1:ml_buf.count)
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
    is_pv_node = (NT === PVNode)
    if hit && !is_singular
        if tt_flag == TT_EXACT
            return tt_score
        elseif tt_flag == TT_LOWER
            α = max(α, tt_score)
        elseif tt_flag == TT_UPPER
            β = min(β, tt_score)
        end
        α ≥ β && return tt_score
    end

    # Internal iterative reduction
    depth -= (tt_best == Move(0) && depth ≥ 3) ? 1 : 0

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
    eval_stack[ply + 1, tid] = eval
    improving = !in_check && ply ≥ 3 && eval > eval_stack[ply - 1, tid]

    # Reverse futility pruning
    if depth ≤ 8 && !in_check && !is_pv_node && !tt_is_pv
        eval ≥ β + (175 * depth - 25 * improving) && return div(eval + β, 2)
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
            move_stack[ply + 1, tid] = (0, 0)
            sc = -negamax(NonPVNode, b, depth - 1 - R, -β, -β + 1, ply + 1, key_history, tid)
            undomove!(b, u)
            sc ≥ β && return sc
        end
    end

    k1  = ply ≤ 256 ? killers[1, ply, tid] : Move(0)
    k2  = ply ≤ 256 ? killers[2, ply, tid] : Move(0)
    sort_moves!(b, ml, ply, tt_best, k1, k2, tid)

    best_score      = -∞
    best_move       = Move(0)
    flag            = TT_UPPER
    searched_quiets = Move[]
    legal_moves     = 0
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

        # Continuation history pruning
        # if !is_pv_node && !in_check && is_quiet && prev_pt > 0
        #     @inbounds Int(cont_hist[to(m).val, cur_pt, prev_to, prev_pt, stm, tid]) < -(MAX_HISTORY ÷ 4) * depth && continue
        # end

        # Singular extension: if the TT move is clearly better than all others,
        # extend it by 1. Prevents recursive singular searches via excluded_move guard.
        ext = 0
        if m == tt_best && !is_singular && depth ≥ 6 && tt_flag ≠ TT_UPPER && ply ≤ 2 * _ROOT_DEPTH[tid] &&
            abs(tt_score) < MATE_SCORE - TT_MAX_PLY && tt_stored_depth ≥ depth - 3

            singular_β  = tt_score - 6 * depth 
            singular_depth = (depth ÷ 2) + 1
            sing_score = negamax(NonPVNode, b, singular_depth, singular_β - 1, singular_β,
                                 ply, key_history, tid; excluded_move = m)
            if sing_score < singular_β
                ext = 1
                if sing_score < singular_β - 40 
                    ext += 1 # double extension
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
        move_stack[ply + 1, tid] = (cur_pt, to(m).val)

        R  = lmr ? lmr_reduction(depth, legal_moves, ischeck(b), is_capture, is_pv_node) : 0

        # # Quiet move futility: if even the maximum reasonable improvement
        # # can't reach alpha, skip this move entirely.
        # if is_quiet && !in_check && !is_pv_node && R > 0 && depth ≤ 8 
        #     lmr_depth = max(new_depth - R, 1)
        #     futility_val = raw_eval + 50 + 150 * Int(best_move == Move(0)) + 150 * lmr_depth
        #     if futility_val ≤ α
        #         pop!(key_history)
        #         undomove!(b, u)
        #         undo_update!(nnue_accs[tid], b, m, nnue_net)
        #         continue
        #     end
        # end

        if i == 1 && is_pv_node
            sc = -negamax(PVNode, b, new_depth, -β, -α, ply + 1, key_history, tid)
        else
            sc = -negamax(NonPVNode, b, new_depth - R, -α - 1, -α, ply + 1, key_history, tid)
            if sc > α
                if R > 0
                    sc = -negamax(NonPVNode, b, new_depth, -α - 1, -α, ply + 1, key_history, tid)
                end
                if is_pv_node && sc > α && sc < β
                    sc = -negamax(PVNode, b, new_depth, -β, -α, ply + 1, key_history, tid)
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

                if α ≥ β
                    flag = TT_LOWER

                    if is_quiet
                        if ply ≤ 256
                            killers[2, ply, tid] = killers[1, ply, tid]
                            killers[1, ply, tid] = m
                        end
                        bonus = depth * depth
                        update_history!(stm, from(m).val, to(m).val, bonus, tid)
                        update_cont_hist!(stm, ply, cur_pt, to(m).val, bonus, tid)
                        for qm in searched_quiets
                            qm_pt = ptype(pieceon(b, from(qm))).val
                            update_history!(stm, from(qm).val, to(qm).val, -bonus, tid)
                            update_cont_hist!(stm, ply, qm_pt, to(qm).val, -bonus, tid)
                        end
                    end
                    break
                end
            end
        end

        is_quiet && push!(searched_quiets, m)
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


function extract_pv_from_tt(root::Board, max_len::Int, key_history::Vector{UInt64})::Vector{Move}
    pv      = Move[]
    bd      = deepcopy(root)
    pv_keys = copy(key_history)
    for _ in 1:max_len
        cnt = 0
        for k in pv_keys
            k == bd.key && (cnt += 1)
        end
        cnt ≥ 2 && break
        _, _, _, _, tt_move, _ = probe_tt(bd.key, 0, 0)
        tt_move == Move(0) && break
        tt_move ∈ moves(bd) || break
        u = domove!(bd, tt_move)
        if was_illegal(bd)
            undomove!(bd, u)
            break
        end
        push!(pv, tt_move)
        push!(pv_keys, bd.key)   # record the reached position, not the old one
        isdraw(bd) && break
    end
    return pv
end

# ============================================================
# Iterative Deepening Root
# ============================================================

function search(b::Board, max_depth::Int, tid::Int)::UInt64
    hv  = @view history[:, :, :, tid]
    cv  = @view cont_hist[:, :, :, :, :, tid]
    cv2 = @view cont_hist2[:, :, :, :, :, tid]
    @inbounds for i in eachindex(hv);  hv[i]  >>= 1; end
    @inbounds for i in eachindex(cv);  cv[i]  >>= 1; end
    @inbounds for i in eachindex(cv2); cv2[i] >>= 1; end
    kv = @view killers[:, :, tid]
    fill!(kv, Move(0))

    refresh!(nnue_accs[tid], b, nnue_net)
    eval_stack[1, tid] = nnue_eval(nnue_accs[tid], b, nnue_net)

    start_ns    = time_ns()
    best_move   = Move(0)
    root_buf    = _MOVE_BUFS[tid][1]
    generate_moves!(root_buf, b)
    ml          = view(root_buf.moves, 1:root_buf.count)
    _NODE_COUNT[tid] = 0
    _SELDEPTH[tid]   = 0
    _ROOT_DEPTH[tid]  = 0
    key_history = copy(game_key_history)
    isempty(ml) && return best_move

    prev_score = 0

    for depth in 1:max_depth
        _ROOT_DEPTH[tid] = depth
        search_stopped[] && break

        window     = 25
        asp_α      = depth ≥ 6 ? prev_score - window : -∞
        asp_β      = depth ≥ 6 ? prev_score + window :  ∞
        depth_best = Move(0)
        best_score = prev_score

        k1 = killers[1, 1, tid]
        k2 = killers[2, 1, tid]
        sort!(ml, by = m -> score_move(b, m, best_move, k1, k2, 1; tid=tid), rev = true)

        while true
            search_stopped[] && break
            α         = asp_α
            iter_best = Move(0)

            root_legal = 0
            for (i, m) in enumerate(ml)
                search_stopped[] && break

                lmr        = root_legal > 1 && depth ≥ 3 && promotion(m) == PieceType(0)
                is_capture = moveiscapture(b, m)

                cur_pt = ptype(pieceon(b, from(m))).val
                update!(nnue_accs[tid], b, m, nnue_net)
                u  = domove!(b, m)
                if was_illegal(b)
                    undomove!(b, u)
                    undo_update!(nnue_accs[tid], b, m, nnue_net)
                    continue
                end
                root_legal += 1
                push!(key_history, b.key)
                move_stack[1, tid] = (cur_pt, to(m).val)

                R  = lmr ? lmr_reduction(depth, root_legal, ischeck(b), is_capture, true) : 0

                if root_legal == 1
                    sc = -negamax(PVNode, b, depth - 1, -asp_β, -α, 1, key_history, tid)
                else
                    sc = -negamax(NonPVNode, b, depth - 1 - R, -α - 1, -α, 1, key_history, tid)
                    if sc > α
                        if R > 0
                            sc = -negamax(NonPVNode, b, depth - 1, -α - 1, -α, 1, key_history, tid)
                        end
                        if sc > α && sc < asp_β
                            sc = -negamax(PVNode, b, depth - 1, -asp_β, -α, 1, key_history, tid)
                        end
                    end
                end

                pop!(key_history)
                undomove!(b, u)
                undo_update!(nnue_accs[tid], b, m, nnue_net)

                if sc > α
                    α         = sc
                    iter_best = m
                    α ≥ asp_β && break
                end
            end

            search_stopped[] && break

            if α ≤ asp_α
                window *= 2
                asp_α = window ≥ 200 ? -∞ : max(asp_α - window, -∞)
            elseif α ≥ asp_β
                iter_best ≠ Move(0) && (depth_best = iter_best; best_score = α)
                window *= 2
                asp_β = window ≥ 200 ?  ∞ : min(asp_β + window, ∞)
            else
                depth_best = iter_best
                best_score = α
                break
            end
        end

        if !search_stopped[] && depth_best ≠ Move(0)
            best_move  = depth_best
            prev_score = best_score
            if tid == 1
                store_tt(b.key, depth, best_score, TT_EXACT, true, depth_best, 0)
                elapsed_ms = div(time_ns() - start_ns, 1_000_000)
                node_count = sum(_NODE_COUNT)
                seldepth   = maximum(_SELDEPTH)
                nps    = elapsed_ms > 0 ? div(node_count * 1000, elapsed_ms) : 0
                pv_str = join(tostring.(extract_pv_from_tt(b, depth, key_history)), " ")
                println("info depth $depth seldepth $seldepth score cp $best_score time $elapsed_ms nodes $node_count nps $nps pv $pv_str")
                flush(stdout)
            end
        end

        time_ns() ≥ search_deadline[] && break
    end

    return best_move
end

# ============================================================
# SMP
# ============================================================

function smp_search(b::Board, max_depth::Int, time_limit::Int)::UInt64
    start_ns = time_ns()
    if time_limit ≥ typemax(Int) >> 20
        search_deadline[] = start_ns + 30_000_000_000
    else
        search_deadline[] = start_ns + UInt64(time_limit) * 1_000_000
    end

    n = _N_THREADS
    if n == 1
        return search(b, max_depth, 1)
    end

    # reset node counts for all threads
    for tid in 1:n
        _NODE_COUNT[tid] = 0
    end

    # threads 2 to n will just populate the TT.
    tasks = Vector{Task}(undef, n)
    for tid in 2:n
        b_copy = deepcopy(b)
        tasks[tid] = Threads.@spawn search(b_copy, max_depth, tid)
    end

    # thread 1 is the master thread that reports the result. 
    result = search(b, max_depth, 1)
    search_stopped[] = true

    for tid in 2:n
        wait(tasks[tid])
    end

    return result
end