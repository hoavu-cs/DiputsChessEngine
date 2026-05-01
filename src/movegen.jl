using Random

# movegen.jl — pseudo-legal move generator, bitboard edition
#
# Squares:    A1=0, B1=1, ... H1=7, A2=8, ... H8=63
#              file = sq & 7    (0..7 = A..H)
#              rank = sq >> 3   (0..7 = 1..8)
#
# Colors:     0 = WHITE, 1 = BLACK
#
# Move encoding:   from | (to << 6) | (promo << 12) | (flag << 16)
#   flags: 0=quiet, 1=double-push, 2=king-castle, 3=queen-castle, 4=en-passant

# ============================================================
# Board — 6 piece-type bitboards × 2 colors
# ============================================================

mutable struct Board
    bb::Vector{UInt64}            # [1:6] white pawn..king, [7:12] black pawn..king
    pieces::Vector{Int8}          # [1:64] sq+1 → bb index (1..12), 0 = empty
    white_bb::UInt64              # all white pieces
    black_bb::UInt64              # all black pieces
    occupied::UInt64              # white_bb | black_bb
    empty::UInt64                 # ~occupied
    stm::Int                      # 0 = WHITE, 1 = BLACK
    castle_rights::Int            # bit: 1=WK, 2=WQ, 4=BK, 8=BQ
    ep_square::Int                # en passant target, 64 = none
    halfmove_clock::Int           # plies since last pawn move or capture
    key::UInt64                   # Zobrist hash
end

const BB_WP = 1;   const BB_BP = 7
const BB_WN = 2;   const BB_BN = 8
const BB_WB = 3;   const BB_BB = 9
const BB_WR = 4;   const BB_BR = 10
const BB_WQ = 5;   const BB_BQ = 11
const BB_WK = 6;   const BB_BK = 12

# ============================================================
# Zobrist hashing
# ============================================================
#
# Flat array layout (same seed as Chess.jl — MersenneTwister(1685)):
#   [1..768]   piece-square:  _zob_psq(bb_idx, sq)
#   [769..776] EP file:       _zob_ep(ep_square)
#   [777..792] castle rights: _zob_castle(rights)  (rights = 0..15)
#   [793]      side to move:  _ZOB_STM  (XOR when stm == 1)
#
# Incremental update in domove!/donullmove!; undomove! restores from UndoInfo.

const _ZOB = let rng = MersenneTwister(1685)
    [rand(rng, UInt64) for _ in 1:793]
end

@inline _zob_psq(bb_idx::Int, sq::Int) = @inbounds _ZOB[(bb_idx - 1) * 64 + sq + 1]
@inline _zob_ep(sq::Int)               = @inbounds _ZOB[768 + file(sq) + 1]
@inline _zob_castle(rights::Int)       = @inbounds _ZOB[776 + rights + 1]
const   _ZOB_STM                       = _ZOB[793]

function compute_key(pos::Board)::UInt64
    k = UInt64(0)
    for bb_idx in 1:12
        bb = pos.bb[bb_idx]
        while bb != 0
            sq, bb = poplsb!(bb)
            k ⊻= _zob_psq(bb_idx, sq)
        end
    end
    pos.stm == 1        && (k ⊻= _ZOB_STM)
    pos.ep_square != 64 && (k ⊻= _zob_ep(pos.ep_square))
    k ⊻= _zob_castle(pos.castle_rights)
    return k
end

# ============================================================
# Move list (preallocated buffer)
# ============================================================

mutable struct MoveList
    moves::Vector{UInt64}
    count::Int
end

MoveList(N = 256) = MoveList(Vector{UInt64}(undef, N), 0)

"""
    Push a move into the move list, given its components:
- `from`: source square (0..63)
- `to`: destination square (0..63)
- `promo`: promotion piece type (0=none, 1=Q, 2=R, 3=B, 4=N)
- `flag`: move flag (0=quiet, 1=double-push, 2=king-castle, 3=queen-castle, 4=en-passant)
"""
@inline function push_move!(ml::MoveList, from::Int, to::Int, promo::Int, flag::Int)
    ml.count += 1
    ml.moves[ml.count] = UInt64(from) | (UInt64(to) << 6) | (UInt64(promo) << 12) | (UInt64(flag) << 16)
end

# ============================================================
# Bit helpers
# ============================================================

@inline file(sq) = sq & 7
@inline rank(sq) = sq >> 3
@inline sq(f, r) = r * 8 + f
@inline bit(sq)  = UInt64(1) << sq

@inline poplsb(bb) = Int(trailing_zeros(bb))
@inline poplsb!(bb::UInt64) = (Int(trailing_zeros(bb)), bb & (bb - 1))

# ============================================================
# Precomputed attack tables
# ============================================================

const KNIGHT_ATTACKS = zeros(UInt64, 64)
const KING_ATTACKS   = zeros(UInt64, 64)
const PAWN_PUSHES    = [zeros(UInt64, 64) for _ in 1:2]  # [color+1][sq+1]
const PAWN_ATTACKS   = [zeros(UInt64, 64) for _ in 1:2]  # [color+1][sq+1]

function init_non_slider_tables!()
    KNIGHT_DELTAS = ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1))

    for s in 0:63
        fr, rr = file(s), rank(s)

        # Knight
        bb = UInt64(0)
        for (df, dr) in KNIGHT_DELTAS
            f, r = fr + df, rr + dr
            0 ≤ f < 8 && 0 ≤ r < 8 && (bb |= bit(sq(f, r)))
        end
        KNIGHT_ATTACKS[s + 1] = bb

        # King
        bb = UInt64(0)
        for df in -1:1, dr in -1:1
            (df, dr) == (0, 0) && continue
            f, r = fr + df, rr + dr
            0 ≤ f < 8 && 0 ≤ r < 8 && (bb |= bit(sq(f, r)))
        end
        KING_ATTACKS[s + 1] = bb

        # White pawn pushes/captures
        PAWN_PUSHES[1][s+1] = (rr < 7) ? bit(s + 8) : UInt64(0)
        bb = UInt64(0)
        fr > 0 && rr < 7 && (bb |= bit(s + 7))
        fr < 7 && rr < 7 && (bb |= bit(s + 9))
        PAWN_ATTACKS[1][s+1] = bb

        # Black pawn pushes/captures
        PAWN_PUSHES[2][s+1] = (rr > 0) ? bit(s - 8) : UInt64(0)
        bb = UInt64(0)
        fr > 0 && rr > 0 && (bb |= bit(s - 9))
        fr < 7 && rr > 0 && (bb |= bit(s - 7))
        PAWN_ATTACKS[2][s+1] = bb
    end
end

# ============================================================
# Move generators
# ============================================================


# Generate pseudo-legal moves for pawns
@inline function gen_pawn!(ml::MoveList, pos::Board, from::Int,
                            color::Int, their_bb::UInt64)
    n = from + 1  # Julia arrays are 1-indexed, squares are 0-indexed

    # Pushes
    push_bb = PAWN_PUSHES[color+1][n]               # precomputed single-push target (0 or 1 bit)
    if push_bb ≠ 0 && push_bb & pos.occupied == 0   
        to = poplsb(push_bb)                     
        # Promoting?
        promo_rank = color == 0 ? 7 : 0            
        if rank(to) == promo_rank
            push_move!(ml, from, to, 1, 0)
            push_move!(ml, from, to, 2, 0)
            push_move!(ml, from, to, 3, 0)
            push_move!(ml, from, to, 4, 0)
        else
            push_move!(ml, from, to, 0, 0)
            # Double push: only from starting rank, and both squares must be empty
            start_rank = color == 0 ? 1 : 6                                 
            if rank(from) == start_rank
                to2 = color == 0 ? from + 16 : from - 16                    
                bb2 = bit(to2)
                bb2 & pos.occupied == 0 && push_move!(ml, from, to2, 0, 1)  
            end
        end
    end

    # Captures
    cap_bb = PAWN_ATTACKS[color+1][n] & their_bb           
    while cap_bb ≠ 0
        to, cap_bb = poplsb!(cap_bb)                           
        promo_rank = color == 0 ? 7 : 0
        if rank(to) == promo_rank
            push_move!(ml, from, to, 1, 0)
            push_move!(ml, from, to, 2, 0)
            push_move!(ml, from, to, 3, 0)
            push_move!(ml, from, to, 4, 0)
        else
            push_move!(ml, from, to, 0, 0)
        end
    end

    # En passant
    pos.ep_square < 64 || return                                    # no ep square set -> done
    PAWN_ATTACKS[color+1][n] & bit(pos.ep_square) == 0 && return    # ep square not attacked
    push_move!(ml, from, pos.ep_square, 0, 4)                       # flag=4 -> en passant
end

# Generate pseudo-legal moves for knights
@inline function gen_knight!(ml::MoveList, from::Int, our_bb::UInt64)
    targets = KNIGHT_ATTACKS[from + 1] & ~our_bb
    while targets != 0
        to, targets = poplsb!(targets)
        push_move!(ml, from, to, 0, 0)
    end
end

# ============================================================
# Magic Bitboards — sliding piece attacks diagram
#
# Example: rook on A1, blockers on D1 and A5
#
#   ROOK_MASKS[A1+1]        occ & ROOK_MASKS[A1+1]    ROOK_TABLE[A1+1][idx]
#   (excl. H1, A8)          (only relevant blockers)   (attack bitboard)
#
#   8 . . . . . . . .       8 . . . . . . . .          8 . . . . . . . .
#   7 x . . . . . . .       7 . . . . . . . .          7 . . . . . . . .
#   6 x . . . . . . .       6 . . . . . . . .          6 . . . . . . . .
#   5 x . . . . . . .       5 1 . . . . . . .          5 x . . . . . . .
#   4 x . . . . . . .       4 . . . . . . . .          4 x . . . . . . .
#   3 x . . . . . . .       3 . . . . . . . .          3 x . . . . . . .
#   2 x . . . . . . .       2 . . . . . . . .          2 x . . . . . . .
#   1 ♖ x x x x x x .      1 . . . 1 . . . .          1 ♖ x x x . . . .
#     A B C D E F G H         A B C D E F G H            A B C D E F G H
#
#   * ROOK_MAGIC >> ROOK_SHIFTS → index into ROOK_TABLE
#   H1/A8 absent from mask: border squares never block further, always reachable
# ============================================================

function _bishop_mask(s::Int)
    bb = UInt64(0)
    fr, rr = file(s), rank(s)
    for (df, dr) in ((-1,-1),(-1,1),(1,-1),(1,1))
        f, r = fr+df, rr+dr
        while 1 ≤ f ≤ 6 && 1 ≤ r ≤ 6
            bb |= bit(sq(f,r)); f += df; r += dr
        end
    end
    return bb
end

function _rook_mask(s::Int)
    fr, rr = file(s), rank(s)
    bb = UInt64(0)
    for f in 1:6; f != fr && (bb |= bit(sq(f, rr))); end
    for r in 1:6; r != rr && (bb |= bit(sq(fr, r))); end
    return bb
end

function _bishop_attacks_slow(s::Int, occ::UInt64)
    bb = UInt64(0)
    fr, rr = file(s), rank(s)
    for (df, dr) in ((-1,-1),(-1,1),(1,-1),(1,1))
        f, r = fr+df, rr+dr
        while 0 ≤ f < 8 && 0 ≤ r < 8
            bb |= bit(sq(f,r))
            (bit(sq(f,r)) & occ) != 0 && break
            f += df; r += dr
        end
    end
    return bb
end

function _rook_attacks_slow(s::Int, occ::UInt64)
    bb = UInt64(0)
    fr, rr = file(s), rank(s)
    for (df, dr) in ((-1,0),(1,0),(0,-1),(0,1))
        f, r = fr+df, rr+dr
        while 0 ≤ f < 8 && 0 ≤ r < 8
            bb |= bit(sq(f,r))
            (bit(sq(f,r)) & occ) != 0 && break
            f += df; r += dr
        end
    end
    return bb
end

# Source: chessprogramming.org/Best_Magics_so_far
const BISHOP_MAGIC = UInt64[
    0x40040844404084,   0x2004208a004208,   0x10190041080202,   0x108060845042010,
    0x581104180800210,  0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
    0x4050404440404,    0x21001420088,      0x24d0080801082102, 0x1020a0a020400,
    0x40308200402,      0x4011002100800,    0x401484104104005,  0x801010402020200,
    0x400210c3880100,   0x404022024108200,  0x810018200204102,  0x4002801a02003,
    0x85040820080400,   0x810102c808880400, 0xe900410884800,    0x8002020480840102,
    0x220200865090201,  0x2010100a02021202, 0x152048408022401,  0x20080002081110,
    0x4001001021004000, 0x800040400a011002, 0xe4004081011002,   0x1c004001012080,
    0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
    0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
    0x209188240001000,  0x400408a884001800, 0x110400a6080400,   0x1840060a44020800,
    0x90080104000041,   0x201011000808101,  0x1a2208080504f080, 0x8012020600211212,
    0x500861011240000,  0x180806108200800,  0x4000020e01040044, 0x300000261044000a,
    0x802241102020002,  0x20906061210001,   0x5a84841004010310, 0x4010801011c04,
    0xa010109502200,    0x4a02012000,       0x500201010098b028, 0x8040002811040900,
    0x28000010020204,   0x6000020202d0240,  0x8918844842082200, 0x4010011029020020,
]

const ROOK_MAGIC = UInt64[
    0x8a80104000800020, 0x140002000100040,  0x2801880a0017001,  0x100081001000420,
    0x200020010080420,  0x3001c0002010008,  0x8480008002000100, 0x2080088004402900,
    0x800098204000,     0x2024401000200040, 0x100802000801000,  0x120800800801000,
    0x208808088000400,  0x2802200800400,    0x2200800100020080, 0x801000060821100,
    0x80044006422000,   0x100808020004000,  0x12108a0010204200, 0x140848010000802,
    0x481828014002800,  0x8094004002004100, 0x4010040010010802, 0x20008806104,
    0x100400080208000,  0x2040002120081000, 0x21200680100081,   0x20100080080080,
    0x2000a00200410,    0x20080800400,      0x80088400100102,   0x80004600042881,
    0x4040008040800020, 0x440003000200801,  0x4200011004500,    0x188020010100100,
    0x14800401802800,   0x2080040080800200, 0x124080204001001,  0x200046502000484,
    0x480400080088020,  0x1000422010034000, 0x30200100110040,   0x100021010009,
    0x2002080100110004, 0x202008004008002,  0x20020004010100,   0x2048440040820001,
    0x101002200408200,  0x40802000401080,   0x4008142004410100, 0x2060820c0120200,
    0x1001004080100,    0x20c020080040080,  0x2935610830022400, 0x44440041009200,
    0x280001040802101,  0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
    0x20030a0244872,    0x12001008414402,   0x2006104900a0804,  0x1004081002402,
]

const BISHOP_MASKS  = zeros(UInt64, 64)
const ROOK_MASKS    = zeros(UInt64, 64)
const BISHOP_SHIFTS = zeros(Int, 64)
const ROOK_SHIFTS   = zeros(Int, 64)
const BISHOP_TABLE  = Vector{Vector{UInt64}}(undef, 64)
const ROOK_TABLE    = Vector{Vector{UInt64}}(undef, 64)

function init_magic_tables!()
    for s in 0:63
        bm = _bishop_mask(s); bn = count_ones(bm)
        rm = _rook_mask(s);   rn = count_ones(rm)
        BISHOP_MASKS[s+1] = bm;  BISHOP_SHIFTS[s+1] = 64 - bn
        ROOK_MASKS[s+1]   = rm;  ROOK_SHIFTS[s+1]   = 64 - rn
        BISHOP_TABLE[s+1] = Vector{UInt64}(undef, 1 << bn)
        ROOK_TABLE[s+1]   = Vector{UInt64}(undef, 1 << rn)
        sub = UInt64(0)
        for _ in 1:(1 << bn)
            idx = Int((sub * BISHOP_MAGIC[s+1]) >> BISHOP_SHIFTS[s+1]) + 1
            BISHOP_TABLE[s+1][idx] = _bishop_attacks_slow(s, sub)
            sub = (sub - bm) & bm
        end
        sub = UInt64(0)
        for _ in 1:(1 << rn)
            idx = Int((sub * ROOK_MAGIC[s+1]) >> ROOK_SHIFTS[s+1]) + 1
            ROOK_TABLE[s+1][idx] = _rook_attacks_slow(s, sub)
            sub = (sub - rm) & rm
        end
    end
end

# Magic lookup: (occ & mask) keeps only the bn relevant blocker bits.
# Multiplying by the magic concentrates them into the top bn bits of the 64-bit product:
#   [ bn bits | 000...000 ]
# Shifting right by (64 - bn) moves them to the bottom:
#   [ 000...000 | bn bits ]  →  index in 0 .. 2^bn - 1
@inline bishop_attacks(s::Int, occ::UInt64) =
    @inbounds BISHOP_TABLE[s+1][Int(((occ & BISHOP_MASKS[s+1]) * BISHOP_MAGIC[s+1]) >> BISHOP_SHIFTS[s+1]) + 1]

# Same magic pattern as bishop_attacks above.
@inline rook_attacks(s::Int, occ::UInt64) =
    @inbounds ROOK_TABLE[s+1][Int(((occ & ROOK_MASKS[s+1]) * ROOK_MAGIC[s+1]) >> ROOK_SHIFTS[s+1]) + 1]

# Castling path masks (squares that must be empty)
const _WK_PATH = bit(5) | bit(6)                    # F1, G1
const _WQ_PATH = bit(1) | bit(2) | bit(3)           # B1, C1, D1
const _BK_PATH = bit(61) | bit(62)                  # F8, G8
const _BQ_PATH = bit(57) | bit(58) | bit(59)        # B8, C8, D8

# Check if a square is attacked by the opponent
@inline function is_attacked(pos::Board, s::Int, by::Int)
    bb  = pos.bb
    occ = pos.occupied
    if by == 0
        (PAWN_ATTACKS[2][s + 1] & bb[BB_WP]) != 0 && return true
        (KNIGHT_ATTACKS[s + 1]  & bb[BB_WN]) != 0 && return true
        (KING_ATTACKS[s + 1]    & bb[BB_WK]) != 0 && return true
        (bishop_attacks(s, occ) & (bb[BB_WB] | bb[BB_WQ])) != 0 && return true
        (rook_attacks(s, occ)   & (bb[BB_WR] | bb[BB_WQ])) != 0 && return true
    else
        (PAWN_ATTACKS[1][s + 1] & bb[BB_BP]) != 0 && return true
        (KNIGHT_ATTACKS[s + 1]  & bb[BB_BN]) != 0 && return true
        (KING_ATTACKS[s + 1]    & bb[BB_BK]) != 0 && return true
        (bishop_attacks(s, occ) & (bb[BB_BB] | bb[BB_BQ])) != 0 && return true
        (rook_attacks(s, occ)   & (bb[BB_BR] | bb[BB_BQ])) != 0 && return true
    end
    return false
end

# After domove!, check if the side that just moved left its king in check.
# b.stm has already flipped, so the moving side is 1 - b.stm.
@inline function was_illegal(b::Board)::Bool
    c        = 1 - b.stm
    king_idx = c == 0 ? BB_WK : BB_BK
    king_bb  = b.bb[king_idx]
    king_bb  == 0 && return true   # own king gone — position is broken
    # Reject king captures: if the side now to move has no king, we just captured it
    opp_king_idx = b.stm == 0 ? BB_WK : BB_BK
    b.bb[opp_king_idx] == 0 && return true
    king_sq = poplsb(king_bb)
    return is_attacked(b, king_sq, b.stm)
end

# Generate pseudo-legal moves for king
@inline function gen_king!(ml::MoveList, pos::Board, from::Int, color::Int, our_bb::UInt64)
    targets = KING_ATTACKS[from + 1] & ~our_bb
    while targets != 0
        to, targets = poplsb!(targets)
        push_move!(ml, from, to, 0, 0)
    end

    occ = pos.occupied
    opp = 1 - color

    if color == 0
        from == 4   || return
        !is_attacked(pos, 4, opp)  || return
        (pos.castle_rights & 1) != 0 && (occ & _WK_PATH) == 0 &&
            (pos.bb[BB_WR] & bit(7)) != 0 &&
            !is_attacked(pos, 5, opp) && !is_attacked(pos, 6, opp) && push_move!(ml, 4, 6, 0, 2)
        (pos.castle_rights & 2) != 0 && (occ & _WQ_PATH) == 0 &&
            (pos.bb[BB_WR] & bit(0)) != 0 &&
            !is_attacked(pos, 3, opp) && !is_attacked(pos, 2, opp) && push_move!(ml, 4, 2, 0, 3)
    else
        from == 60  || return
        !is_attacked(pos, 60, opp) || return
        (pos.castle_rights & 4) != 0 && (occ & _BK_PATH) == 0 &&
            (pos.bb[BB_BR] & bit(63)) != 0 &&
            !is_attacked(pos, 61, opp) && !is_attacked(pos, 62, opp) && push_move!(ml, 60, 62, 0, 2)
        (pos.castle_rights & 8) != 0 && (occ & _BQ_PATH) == 0 &&
            (pos.bb[BB_BR] & bit(56)) != 0 &&
            !is_attacked(pos, 59, opp) && !is_attacked(pos, 58, opp) && push_move!(ml, 60, 58, 0, 3)
    end
end

@inline function gen_bishop!(ml::MoveList, pos::Board, from::Int, our_bb::UInt64)
    targets = bishop_attacks(from, pos.occupied) & ~our_bb
    while targets != 0
        to, targets = poplsb!(targets)
        push_move!(ml, from, to, 0, 0)
    end
end

@inline function gen_rook!(ml::MoveList, pos::Board, from::Int, our_bb::UInt64)
    targets = rook_attacks(from, pos.occupied) & ~our_bb
    while targets != 0
        to, targets = poplsb!(targets)
        push_move!(ml, from, to, 0, 0)
    end
end

# ============================================================
# Main generator
# ============================================================

function generate_moves!(ml::MoveList, pos::Board)
    ml.count = 0
    bb    = pos.bb
    c     = pos.stm
    base  = c * 6
    our   = c == 0 ? pos.white_bb : pos.black_bb
    their = c == 0 ? pos.black_bb : pos.white_bb

    pcs = bb[base + 1]
    while pcs != 0; from, pcs = poplsb!(pcs); gen_pawn!(ml, pos, from, c, their); end

    pcs = bb[base + 2]
    while pcs != 0; from, pcs = poplsb!(pcs); gen_knight!(ml, from, our); end

    pcs = bb[base + 3]
    while pcs != 0; from, pcs = poplsb!(pcs); gen_bishop!(ml, pos, from, our); end

    pcs = bb[base + 4]
    while pcs != 0; from, pcs = poplsb!(pcs); gen_rook!(ml, pos, from, our); end

    pcs = bb[base + 5]
    while pcs != 0
        from, pcs = poplsb!(pcs)
        gen_bishop!(ml, pos, from, our)
        gen_rook!(ml, pos, from, our)
    end

    pcs = bb[base + 6]
    while pcs != 0; from, pcs = poplsb!(pcs); gen_king!(ml, pos, from, c, our); end
end

function generate_moves(pos::Board)::MoveList
    ml = MoveList()
    generate_moves!(ml, pos)
    return ml
end

# ============================================================
# Board queries
# ============================================================

function isdraw(pos::Board)::Bool
    pos.halfmove_clock >= 100 && return true

    # Pawns, rooks, queens → not drawn by material
    (pos.bb[BB_WP] | pos.bb[BB_BP] |
     pos.bb[BB_WR] | pos.bb[BB_BR] |
     pos.bb[BB_WQ] | pos.bb[BB_BQ]) != 0 && return false

    wn = count_ones(pos.bb[BB_WN]); wb = count_ones(pos.bb[BB_WB])
    bn = count_ones(pos.bb[BB_BN]); bb = count_ones(pos.bb[BB_BB])
    wm = wn + wb;  bm = bn + bb

    # K vs K
    wm == 0 && bm == 0 && return true
    # K+N vs K  or  K+B vs K  (single minor either side)
    (wm == 1 && bm == 0) && return true
    (wm == 0 && bm == 1) && return true
    # K+minor vs K+minor — one each side, never sufficient to force mate
    (wm == 1 && bm == 1) && return true
    # K+N+N vs K  (two knights can't force mate against bare king)
    (wm == 2 && bm == 0 && wn == 2) && return true
    (wm == 0 && bm == 2 && bn == 2) && return true

    return false
end

@inline function ischeck(pos::Board)::Bool
    king_idx = pos.stm == 0 ? BB_WK : BB_BK
    king_bb  = pos.bb[king_idx]
    king_bb  == 0 && return false
    king_sq  = poplsb(king_bb)
    return is_attacked(pos, king_sq, 1 - pos.stm)
end

@inline function moveiscapture(pos::Board, move::UInt64)::Bool
    Int((move >> 16) & 0xf) == 4 && return true   # en passant
    to_bb    = bit(Int((move >> 6) & 0x3f))
    their_bb = pos.stm == 0 ? pos.black_bb : pos.white_bb
    return (their_bb & to_bb) != 0
end

# ============================================================
# Make / Unmake
# ============================================================

# Castle rights mask tables: &= MASK_FROM[from+1] & MASK_TO[to+1]
const _CASTLE_MASK_FROM = let m = fill(0x0f, 64)
    m[1]  = 0x0d   # A1: clear WQ
    m[5]  = 0x0c   # E1: white king moved — clear WK+WQ
    m[8]  = 0x0e   # H1: clear WK
    m[57] = 0x07   # A8: clear BQ
    m[61] = 0x03   # E8: black king moved — clear BK+BQ
    m[64] = 0x0b   # H8: clear BK
    Tuple(m)
end
const _CASTLE_MASK_TO = let m = fill(0x0f, 64)
    m[1]  = 0x0d   # A1 rook captured: clear WQ
    m[8]  = 0x0e   # H1 rook captured: clear WK
    m[57] = 0x07   # A8 rook captured: clear BQ
    m[64] = 0x0b   # H8 rook captured: clear BK
    Tuple(m)
end

# promo encoding: 1=Q,2=R,3=B,4=N → bb offset within color
# matches BB_WQ=5, BB_WR=4, BB_WB=3, BB_WN=2
const _PROMO_OFFSET = (5, 4, 3, 2)

struct UndoInfo
    move::UInt64
    captured_bb::Int   # pos.bb index of captured piece, 0 = none
    castle_rights::Int
    ep_square::Int
    halfmove_clock::Int
    key::UInt64
end

function domove!(pos::Board, move::UInt64)::UndoInfo
    from  = Int(move         & 0x3f)
    to    = Int((move >>  6) & 0x3f)
    promo = Int((move >> 12) & 0xf)
    flag  = Int((move >> 16) & 0xf)

    c    = pos.stm
    opp  = 1 - c
    base = c * 6       

    src_bb = bit(from)
    dst_bb = bit(to)

    # Find moving and captured pieces via piece map (O(1))
    moving_idx = Int(pos.pieces[from + 1])
    @assert moving_idx != 0 "domove!: no piece at from=$from (stm=$(pos.stm))"
    opp_base = opp * 6
    captured_idx = flag != 4 ? Int(pos.pieces[to + 1]) : 0

    undo = UndoInfo(move, captured_idx, pos.castle_rights, pos.ep_square, pos.halfmove_clock, pos.key)

    # === Zobrist: XOR out old state ===
    k = pos.key
    k ⊻= _zob_psq(moving_idx, from)
    k ⊻= _zob_castle(pos.castle_rights)
    pos.ep_square != 64 && (k ⊻= _zob_ep(pos.ep_square))
    flag != 4 && captured_idx != 0 && (k ⊻= _zob_psq(captured_idx, to))
    if flag == 2
        c == 0 ? (k ⊻= _zob_psq(BB_WR, 7))  : (k ⊻= _zob_psq(BB_BR, 63))
    elseif flag == 3
        c == 0 ? (k ⊻= _zob_psq(BB_WR, 0))  : (k ⊻= _zob_psq(BB_BR, 56))
    end

    # Remove captured piece
    captured_idx != 0 && (pos.bb[captured_idx] &= ~dst_bb)

    # Move piece: clear src, set dst
    pos.bb[moving_idx] = (pos.bb[moving_idx] & ~src_bb) | dst_bb
    pos.pieces[from + 1] = Int8(0)
    pos.pieces[to + 1]   = Int8(moving_idx)

    # Special flags
    if flag == 1      # double pawn push — set ep square
        pos.ep_square = c == 0 ? to - 8 : to + 8
    elseif flag == 4  # en passant — remove captured pawn
        ep_sq = c == 0 ? to - 8 : to + 8
        pos.bb[opp_base + 1] &= ~bit(ep_sq)   # opp pawn bb = opp_base + BB_WP(1)
        pos.pieces[ep_sq + 1] = Int8(0)
        pos.ep_square = 64
    elseif flag == 2  # kingside castle — move rook
        if c == 0
            pos.bb[BB_WR] = (pos.bb[BB_WR] & ~bit(7))  | bit(5)
            pos.pieces[8] = Int8(0); pos.pieces[6] = Int8(BB_WR)
        else
            pos.bb[BB_BR] = (pos.bb[BB_BR] & ~bit(63)) | bit(61)
            pos.pieces[64] = Int8(0); pos.pieces[62] = Int8(BB_BR)
        end
        pos.ep_square = 64
    elseif flag == 3  # queenside castle — move rook
        if c == 0
            pos.bb[BB_WR] = (pos.bb[BB_WR] & ~bit(0))  | bit(3)
            pos.pieces[1] = Int8(0); pos.pieces[4] = Int8(BB_WR)
        else
            pos.bb[BB_BR] = (pos.bb[BB_BR] & ~bit(56)) | bit(59)
            pos.pieces[57] = Int8(0); pos.pieces[60] = Int8(BB_BR)
        end
        pos.ep_square = 64
    else
        pos.ep_square = 64
    end

    # Promotion — replace pawn at dst with promoted piece
    if promo != 0
        pos.bb[base + 1]                    &= ~dst_bb   # remove pawn
        pos.bb[base + _PROMO_OFFSET[promo]]  |= dst_bb   # add promoted piece
        pos.pieces[to + 1] = Int8(base + _PROMO_OFFSET[promo])
    end

    # Update castle rights (2 table lookups replace 10 branches)
    pos.castle_rights &= _CASTLE_MASK_FROM[from + 1] & _CASTLE_MASK_TO[to + 1]

    # === Zobrist: XOR in new state (ep_square and castle_rights already updated) ===
    promo != 0 ? (k ⊻= _zob_psq(base + _PROMO_OFFSET[promo], to)) :
                 (k ⊻= _zob_psq(moving_idx, to))
    if flag == 4
        k ⊻= _zob_psq(opp_base + 1, c == 0 ? to - 8 : to + 8)
    elseif flag == 2
        c == 0 ? (k ⊻= _zob_psq(BB_WR, 5))  : (k ⊻= _zob_psq(BB_BR, 61))
    elseif flag == 3
        c == 0 ? (k ⊻= _zob_psq(BB_WR, 3))  : (k ⊻= _zob_psq(BB_BR, 59))
    end
    k ⊻= _zob_castle(pos.castle_rights)
    pos.ep_square != 64 && (k ⊻= _zob_ep(pos.ep_square))
    k ⊻= _ZOB_STM

    if flag == 4        # en passant
        ep_sq  = c == 0 ? to - 8 : to + 8
        ep_bit = bit(ep_sq)
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb; pos.black_bb ⊻= ep_bit
        else;      pos.black_bb ⊻= src_bb | dst_bb; pos.white_bb ⊻= ep_bit; end
    elseif flag == 2    # kingside castle: king + rook
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb | bit(7) | bit(5)
        else;      pos.black_bb ⊻= src_bb | dst_bb | bit(63) | bit(61); end
    elseif flag == 3    # queenside castle
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb | bit(0) | bit(3)
        else;      pos.black_bb ⊻= src_bb | dst_bb | bit(56) | bit(59); end
    elseif captured_idx != 0  # normal capture (includes promotion-capture)
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb; pos.black_bb ⊻= dst_bb
        else;      pos.black_bb ⊻= src_bb | dst_bb; pos.white_bb ⊻= dst_bb; end
    else                # quiet / double push / promotion (no capture)
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb
        else;      pos.black_bb ⊻= src_bb | dst_bb; end
    end
    pos.occupied = pos.white_bb | pos.black_bb
    pos.empty    = ~pos.occupied

    pos.halfmove_clock = (captured_idx != 0 || flag == 4 || moving_idx == BB_WP || moving_idx == BB_BP) ? 0 : pos.halfmove_clock + 1
    pos.stm      = opp
    pos.key      = k

    return undo
end

function donullmove!(pos::Board)::UndoInfo
    undo = UndoInfo(UInt64(0), 0, pos.castle_rights, pos.ep_square, pos.halfmove_clock, pos.key)
    k = pos.key
    pos.ep_square != 64 && (k ⊻= _zob_ep(pos.ep_square))
    k ⊻= _ZOB_STM
    pos.ep_square      = 64
    pos.halfmove_clock += 1
    pos.stm            = 1 - pos.stm
    pos.key            = k
    return undo
end

function undomove!(pos::Board, u::UndoInfo)
    pos.key = u.key
    if u.move == UInt64(0)
        pos.stm            = 1 - pos.stm
        pos.castle_rights  = u.castle_rights
        pos.ep_square      = u.ep_square
        pos.halfmove_clock = u.halfmove_clock
        return
    end

    move  = u.move
    from  = Int(move         & 0x3f)
    to    = Int((move >>  6) & 0x3f)
    promo = Int((move >> 12) & 0xf)
    flag  = Int((move >> 16) & 0xf)

    pos.stm = 1 - pos.stm
    c    = pos.stm
    opp  = 1 - c
    base = c * 6

    src_bb = bit(from)
    dst_bb = bit(to)

    if promo != 0
        # Undo promotion: remove promoted piece, restore pawn at src
        pos.bb[base + _PROMO_OFFSET[promo]] &= ~dst_bb
        pos.bb[base + 1]                     |= src_bb
        pos.pieces[to + 1]   = Int8(u.captured_bb)
        pos.pieces[from + 1] = Int8(base + 1)
    else
        # Move piece back: dst → src via piece map (O(1))
        moving_idx = Int(pos.pieces[to + 1])
        pos.bb[moving_idx] = (pos.bb[moving_idx] & ~dst_bb) | src_bb
        pos.pieces[to + 1]   = Int8(u.captured_bb)
        pos.pieces[from + 1] = Int8(moving_idx)
    end

    # Restore captured piece
    u.captured_bb != 0 && (pos.bb[u.captured_bb] |= dst_bb)

    # Restore en passant pawn
    if flag == 4
        ep_sq = c == 0 ? to - 8 : to + 8
        pos.bb[opp * 6 + 1] |= bit(ep_sq)   # opp pawn
        pos.pieces[ep_sq + 1] = Int8(opp * 6 + 1)
    end

    # Restore castling rook
    if flag == 2
        if c == 0
            pos.bb[BB_WR] = (pos.bb[BB_WR] & ~bit(5))  | bit(7)
            pos.pieces[6] = Int8(0); pos.pieces[8] = Int8(BB_WR)
        else
            pos.bb[BB_BR] = (pos.bb[BB_BR] & ~bit(61)) | bit(63)
            pos.pieces[62] = Int8(0); pos.pieces[64] = Int8(BB_BR)
        end
    elseif flag == 3
        if c == 0
            pos.bb[BB_WR] = (pos.bb[BB_WR] & ~bit(3))  | bit(0)
            pos.pieces[4] = Int8(0); pos.pieces[1] = Int8(BB_WR)
        else
            pos.bb[BB_BR] = (pos.bb[BB_BR] & ~bit(59)) | bit(56)
            pos.pieces[60] = Int8(0); pos.pieces[57] = Int8(BB_BR)
        end
    end

    pos.castle_rights  = u.castle_rights
    pos.ep_square      = u.ep_square
    pos.halfmove_clock = u.halfmove_clock

    if flag == 4        # en passant (XOR is self-inverse — same as domove!)
        ep_sq  = c == 0 ? to - 8 : to + 8
        ep_bit = bit(ep_sq)
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb; pos.black_bb ⊻= ep_bit
        else;      pos.black_bb ⊻= src_bb | dst_bb; pos.white_bb ⊻= ep_bit; end
    elseif flag == 2    # kingside castle
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb | bit(7) | bit(5)
        else;      pos.black_bb ⊻= src_bb | dst_bb | bit(63) | bit(61); end
    elseif flag == 3    # queenside castle
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb | bit(0) | bit(3)
        else;      pos.black_bb ⊻= src_bb | dst_bb | bit(56) | bit(59); end
    elseif u.captured_bb != 0  # normal capture
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb; pos.black_bb ⊻= dst_bb
        else;      pos.black_bb ⊻= src_bb | dst_bb; pos.white_bb ⊻= dst_bb; end
    else                # quiet / double push / promotion (no capture)
        if c == 0; pos.white_bb ⊻= src_bb | dst_bb
        else;      pos.black_bb ⊻= src_bb | dst_bb; end
    end
    pos.occupied = pos.white_bb | pos.black_bb
    pos.empty    = ~pos.occupied
end

# ============================================================
# Types and wrappers matching Chess.jl interface
# ============================================================

const Move = UInt64

struct Square;    val::Int; end   # 1-indexed, used as array index in history tables
struct PieceType; val::Int; end   # 0=none, 1=P..6=K
struct Piece;     val::Int; end   # 1-12 = bb index, 0 = empty square

Base.:(==)(a::PieceType, b::PieceType) = a.val == b.val
Base.:(==)(a::PieceType, b::Int)      = a.val == b
Base.:(==)(a::Int,       b::PieceType) = a == b.val
Base.:(≠)(a::PieceType, b::PieceType) = a.val != b.val
Base.:(≠)(a::PieceType, b::Int)       = a.val != b

const WHITE  = 0;  const BLACK  = 1
const PAWN   = 1;  const KNIGHT = 2;  const BISHOP = 3
const ROOK   = 4;  const QUEEN  = 5;  const KING   = 6

# Move decoders — .val is 1-indexed so it can directly index 64-element arrays
from(m::UInt64)        = Square(Int(m & 0x3f) + 1)
to(m::UInt64)          = Square(Int((m >> 6) & 0x3f) + 1)
function promotion(m::UInt64)::PieceType
    p = Int((m >> 12) & 0xf)
    p == 0 && return PieceType(0)
    p == 1 && return PieceType(QUEEN)
    p == 2 && return PieceType(ROOK)
    p == 3 && return PieceType(BISHOP)
    return PieceType(KNIGHT)
end

sidetomove(b::Board) = b.stm   # returns WHITE(0) or BLACK(1)

# Piece on a square (sq.val is 1-indexed)
pieceon(b::Board, sq::Square) = Piece(b.pieces[sq.val])

# Piece type 1-6 from a Piece (works for both colors; bb index mod 6)
ptype(p::Piece)  = PieceType(p.val == 0 ? 0 : ((p.val - 1) % 6) + 1)
pcolor(p::Piece) = p.val == 0 ? -1 : (p.val <= 6 ? WHITE : BLACK)

# Iterable over squares in a bitboard — yields Square structs
struct Squares; bb::UInt64; end
Base.iterate(s::Squares, bb = s.bb) =
    bb == 0 ? nothing : let idx = poplsb(bb); (Square(idx + 1), bb & (bb - 1)) end
Base.isempty(s::Squares) = s.bb == 0

pieces(b::Board, side::Int, pt::Int) = Squares(b.bb[side * 6 + pt])

# Return a view over the generated pseudo-legal moves
moves(b::Board) = (ml = generate_moves(b); view(ml.moves, 1:ml.count))

ischeckmate(b::Board) = ischeck(b)  && isempty(moves(b))
isstalemate(b::Board) = !ischeck(b) && isempty(moves(b))

function tostring(m::UInt64)::String
    m == Move(0) && return "0000"
    fsq   = Int(m & 0x3f)
    tsq   = Int((m >> 6) & 0x3f)
    promo = Int((m >> 12) & 0xf)
    s = string(Char('a' + file(fsq)), rank(fsq) + 1,
               Char('a' + file(tsq)), rank(tsq) + 1)
    promo != 0 && (s *= string("qrbn"[promo]))
    return s
end

import Base.deepcopy
Base.deepcopy(b::Board) =
    Board(copy(b.bb), copy(b.pieces), b.white_bb, b.black_bb, b.occupied, b.empty, b.stm, b.castle_rights, b.ep_square, b.halfmove_clock, b.key)

function from_fen(fen::String)::Board
    bb   = zeros(UInt64, 12)
    toks = split(fen)

    # Piece placement
    c, r = 0, 7
    for ch in toks[1]
        if ch == '/'
            r -= 1; c = 0
        elseif isdigit(ch)
            c += parse(Int, ch)
        else
            idx = findfirst(==( ch), "PNBRQKpnbrqk")
            idx !== nothing && (bb[idx] |= bit(r * 8 + c); c += 1)
        end
    end

    stm = length(toks) >= 2 && toks[2] == "b" ? BLACK : WHITE

    castle_rights = 0
    if length(toks) >= 3 && toks[3] != "-"
        'K' in toks[3] && (castle_rights |= 1)
        'Q' in toks[3] && (castle_rights |= 2)
        'k' in toks[3] && (castle_rights |= 4)
        'q' in toks[3] && (castle_rights |= 8)
    end

    ep_square = 64
    if length(toks) >= 4 && toks[4] != "-"
        ep_f = toks[4][1] - 'a'
        ep_r = parse(Int, toks[4][2]) - 1
        ep_square = ep_r * 8 + ep_f
    end

    halfmove_clock = length(toks) >= 5 ? parse(Int, toks[5]) : 0

    pm = zeros(Int8, 64)
    for i in 1:12
        b2 = bb[i]
        while b2 != 0
            s, b2 = poplsb!(b2)
            pm[s + 1] = Int8(i)
        end
    end

    white_bb = bb[1]|bb[2]|bb[3]|bb[4]|bb[5]|bb[6]
    black_bb = bb[7]|bb[8]|bb[9]|bb[10]|bb[11]|bb[12]
    occ = white_bb | black_bb
    pos = Board(bb, pm, white_bb, black_bb, occ, ~occ, stm, castle_rights, ep_square, halfmove_clock, UInt64(0))
    pos.key = compute_key(pos)
    pos
end

# Find the legal move matching a UCI string (e.g. "e2e4", "e7e8q")
function parse_move(b::Board, s::AbstractString)::UInt64
    length(s) < 4 && return Move(0)
    fsq = (s[2] - '1') * 8 + (s[1] - 'a')
    tsq = (s[4] - '1') * 8 + (s[3] - 'a')
    promo = length(s) >= 5 ? something(findfirst(==(s[5]), "qrbn"), 0) : 0
    for m in moves(b)
        Int(m & 0x3f) == fsq &&
        Int((m >> 6) & 0x3f) == tsq &&
        Int((m >> 12) & 0xf) == promo && return m
    end
    return Move(0)
end

# ============================================================
# Initialization (runs at include time)
# ============================================================

init_non_slider_tables!()
init_magic_tables!()
