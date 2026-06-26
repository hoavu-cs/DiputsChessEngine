using Base.Threads
include("movegen.jl")

const ENGINE_NAME   = "Diputs Chess Engine"
const ENGINE_AUTHOR = "Hoa T. Vu"
const DEV_MODE      = false   # false = hide tunable search-param UCI options (release)

include("openings.jl")

const _BOOK = Dict{UInt64, Vector{String}}()

function _init_book!()
    startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for line in _BOOK_LINES
        b = from_fen(startfen)
        for ms in line
            key = b.key
            m = parse_move(b, ms)
            m == Move(0) && break
            if !haskey(_BOOK, key)
                _BOOK[key] = String[]
            end
            ms ∉ _BOOK[key] && push!(_BOOK[key], ms)
            domove!(b, m)
        end
    end
end

# ============================================================
# Global State
# ============================================================

board                        = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
game_key_history             = UInt64[board.key]
max_depth::Int               = 99
use_own_book::Bool           = false
search_stopped::Atomic{Bool} = Atomic{Bool}(false)
search_running::Atomic{Bool} = Atomic{Bool}(false)

# ============================================================
# Position & Options
# ============================================================

function process_position(command::String)
    tokens = String.(split(command))
    idx = 2
    if idx > length(tokens)
        return
    end

    if tokens[idx] == "startpos"
        global board = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        idx += 1
    elseif tokens[idx] == "fen"
        fen_parts = String[]
        idx += 1
        while idx ≤ length(tokens) && tokens[idx] ≠ "moves"
            push!(fen_parts, tokens[idx])
            idx += 1
        end
        fen_str = join(fen_parts, " ")
        global board = from_fen(fen_str)
    else
        return
    end

    global game_key_history = sizehint!(UInt64[board.key], 512)

    if idx ≤ length(tokens) && tokens[idx] == "moves"
        idx += 1
        while idx ≤ length(tokens)
            move = parse_move(board, tokens[idx])
            if move ≠ Move(0)
                domove!(board, move)
                push!(game_key_history, board.key)
            end
            idx += 1
        end
    end
end

function process_option(tokens::Vector{String})
    if length(tokens) < 5
        return
    end
    option_name = tokens[3]
    value_str   = tokens[5]
    if option_name == "Depth"
        global max_depth = parse(Int, value_str)
    elseif option_name == "Hash"
        resize_tt(parse(Int, value_str))
    elseif option_name == "OpeningBook"
        global use_own_book = value_str == "true"
    end
end

include("nnue.jl")
include("params.jl")
include("search.jl")
_init_book!()

# ============================================================
# Search Thread
# ============================================================

function search_thread(search_board::Board, search_depth::Int, time_soft::Int, time_hard::Int)
    try
        best_move = smp_search(search_board, search_depth, time_soft, time_hard)
        println(best_move ≠ Move(0) ? "bestmove $(tostring(best_move))" : "bestmove 0000")
        flush(stdout)
    catch e
        println("info string ERROR: $e")
        println("bestmove 0000")
        flush(stdout)
    finally
        search_running[] = false
    end
end

function process_go(tokens::Vector{String})
    if use_own_book && haskey(_BOOK, board.key)
        candidates = filter(ms -> parse_move(board, ms) ≠ Move(0), _BOOK[board.key])
        if !isempty(candidates)
            println("bestmove $(rand(candidates))")
            flush(stdout)
            return
        end
    end

    search_stopped[] = false
    search_running[] = true

    time_soft     = 30000
    time_hard     = 30000
    search_depth  = max_depth
    depth_limited = false
    wtime = btime = winc = binc = movestogo = movetime = 0

    idx = 2
    while idx ≤ length(tokens)
        if tokens[idx] == "wtime" && idx + 1 ≤ length(tokens)
            wtime = parse(Int, tokens[idx + 1]); idx += 2
        elseif tokens[idx] == "btime" && idx + 1 ≤ length(tokens)
            btime = parse(Int, tokens[idx + 1]); idx += 2
        elseif tokens[idx] == "winc" && idx + 1 ≤ length(tokens)
            winc = parse(Int, tokens[idx + 1]); idx += 2
        elseif tokens[idx] == "binc" && idx + 1 ≤ length(tokens)
            binc = parse(Int, tokens[idx + 1]); idx += 2
        elseif tokens[idx] == "movestogo" && idx + 1 ≤ length(tokens)
            movestogo = parse(Int, tokens[idx + 1]); idx += 2
        elseif tokens[idx] == "movetime" && idx + 1 ≤ length(tokens)
            movetime = parse(Int, tokens[idx + 1]); idx += 2
        elseif tokens[idx] == "depth" && idx + 1 ≤ length(tokens)
            search_depth  = parse(Int, tokens[idx + 1])
            depth_limited = true
            time_soft = time_hard = typemax(Int)
            idx += 2
        else
            idx += 1
        end
    end

    if !depth_limited
        if movetime > 0
            time_soft = time_hard = movetime
        else
            my_time = sidetomove(board) == WHITE ? wtime : btime
            my_inc  = sidetomove(board) == WHITE ? winc  : binc
            if my_time > 0
                overhead  = 50
                time_soft = clamp(div(my_time, 25) + div(my_inc, 2), 0, div(my_time, 2))
                time_hard = min(time_soft * 3, my_time ÷ 2 - overhead)
            end
        end
    end

    Threads.@spawn search_thread(deepcopy(board), search_depth, time_soft, time_hard)
end

function process_stop()
    search_running[] && (search_stopped[] = true)
end

# ============================================================
# Bench crap
# ============================================================

const _BENCH_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "2r5/3pk3/8/1P6/2K5/8/8/8 w - - 5 4",
    "rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
    "r2q1rk1/ppp2ppp/2n1bn2/3pp3/1bBPP3/2N1BN2/PPP2PPP/R2QR1K1 w - - 4 9",
    "r4rk1/pp3ppp/1qp1bn2/3p4/3P4/1QN1BN2/PP3PPP/R4RK1 w - - 2 14",
    "8/8/1p6/3b4/1P1k4/3B4/8/4K3 b - - 0 1",
    "8/8/7p/3KNN1k/2p4p/8/3P2p1/8 w - - 0 1",
    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
    "8/8/4k3/8/2p5/8/B2P4/4K3 w - - 0 1",
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
]

function run_bench(depth::Int)
    total_nodes = 0
    t0 = time_ns()
    for fen in _BENCH_POSITIONS
        b = from_fen(fen)
        clear_tt()
        clear_history()
        search_stopped[] = false
        smp_search(b, depth, typemax(Int), typemax(Int))
        nodes = sum(_NODE_COUNT)
        total_nodes += nodes
        println("info string $(fen[1:min(40,end)])... nodes $nodes")
        flush(stdout)
    end
    elapsed_ms = (time_ns() - t0) ÷ 1_000_000
    nps = elapsed_ms > 0 ? total_nodes * 1000 ÷ elapsed_ms : 0
    println("$total_nodes nodes $(nps) nps")
    flush(stdout)
end

# ============================================================
# UCI Loop
# ============================================================

function process_uci()
    println("id name $(ENGINE_NAME)")
    println("id author $(ENGINE_AUTHOR)")
    println("option name Depth type spin default 99 min 1 max 99")
    println("option name Hash type spin default 256 min 64 max 1024")
    println("option name OpeningBook type check default false")
    println("uciok")
    flush(stdout)
end

function uci_loop()
    while true
        try
            line = String(strip(readline()))
            isempty(line) && continue

            if line == "uci"
                process_uci()
            elseif line == "isready"
                println("readyok"); flush(stdout)
            elseif line == "ucinewgame"
                global board = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                clear_tt()
                clear_history()
            elseif startswith(line, "position")
                process_position(line)
            elseif startswith(line, "setoption")
                process_option(String.(split(line)))
            elseif startswith(line, "go")
                process_go(String.(split(line)))
            elseif line == "stop"
                process_stop()
            elseif startswith(line, "bench")
                tokens = split(line)
                depth = length(tokens) ≥ 2 ? parse(Int, tokens[2]) : 12
                run_bench(depth)
            elseif line == "quit"
                search_stopped[] = true
                break
            end
        catch e
            println(stderr, "UCI loop error: $e")
        end
    end
end

function julia_main()::Cint
    uci_loop()
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    uci_loop()
end
