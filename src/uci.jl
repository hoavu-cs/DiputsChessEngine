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

    global game_key_history = UInt64[board.key]

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
    elseif DEV_MODE && option_name == "RFP_Margin"
        sp.rfp_margin = parse(Int, value_str)
    elseif DEV_MODE && option_name == "RFP_Improving"
        sp.rfp_improving = parse(Int, value_str)
    elseif DEV_MODE && option_name == "RFP_MaxDepth"
        sp.rfp_max_depth = parse(Int, value_str)
    elseif DEV_MODE && option_name == "NMP_Base"
        sp.nmp_base = parse(Int, value_str)
    elseif DEV_MODE && option_name == "NMP_DepthDiv"
        sp.nmp_depth_div = parse(Int, value_str)
    elseif DEV_MODE && option_name == "NMP_EvalDiv"
        sp.nmp_eval_div = parse(Int, value_str)
    elseif DEV_MODE && option_name == "NMP_MaxExtra"
        sp.nmp_max_extra = parse(Int, value_str)
    elseif DEV_MODE && option_name == "NMP_MinDepth"
        sp.nmp_min_depth = parse(Int, value_str)
    elseif DEV_MODE && option_name == "LMP_Base"
        sp.lmp_base = parse(Int, value_str)
    elseif DEV_MODE && option_name == "LMP_Div"
        sp.lmp_div = parse(Int, value_str)
    elseif DEV_MODE && option_name == "LMP_MaxDepth"
        sp.lmp_max_depth = parse(Int, value_str)
    elseif DEV_MODE && option_name == "LMR_Divisor"
        sp.lmr_divisor = parse(Int, value_str)
        init_lmr_table!()
    elseif DEV_MODE && option_name == "LMR_InCheck"
        sp.lmr_in_check = parse(Int, value_str)
    elseif DEV_MODE && option_name == "LMR_Capture"
        sp.lmr_capture = parse(Int, value_str)
    elseif DEV_MODE && option_name == "LMR_NonPV"
        sp.lmr_non_pv = parse(Int, value_str)
    elseif DEV_MODE && option_name == "Asp_Window"
        sp.asp_window = parse(Int, value_str)
    elseif DEV_MODE && option_name == "Asp_Depth"
        sp.asp_depth = parse(Int, value_str)
    elseif DEV_MODE && option_name == "Asp_Max"
        sp.asp_max = parse(Int, value_str)
    elseif DEV_MODE && option_name == "IIR_MinDepth"
        sp.iir_min_depth = parse(Int, value_str)
    elseif DEV_MODE && option_name == "SE_Margin"
        sp.se_margin = parse(Int, value_str)
    elseif DEV_MODE && option_name == "SE_MinDepth"
        sp.se_min_depth = parse(Int, value_str)
    elseif DEV_MODE && option_name == "SE_DoubleMargin"
        sp.se_double_margin = parse(Int, value_str)
    elseif DEV_MODE && option_name == "MaxHistory"
        sp.max_history = parse(Int, value_str)
    elseif DEV_MODE && option_name == "Corr_Delta"
        sp.corr_Δ = parse(Int, value_str)
    elseif DEV_MODE && option_name == "Corr_Learning"
        sp.corr_δ = parse(Int, value_str)
    elseif DEV_MODE && option_name == "Corr_Gamma"
        sp.corr_Γ = parse(Int, value_str)
    end
end

include("nnue.jl")
include("search.jl")
_init_book!()

# ============================================================
# Search Thread
# ============================================================

function search_thread(search_board::Board, search_depth::Int, time_limit::Int)
    try
        best_move = smp_search(search_board, search_depth, time_limit)
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

    time_limit    = 30000
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
            time_limit    = typemax(Int)
            idx += 2
        else
            idx += 1
        end
    end

    if !depth_limited
        if movetime > 0
            time_limit = movetime
        else
            my_time = sidetomove(board) == WHITE ? wtime : btime
            my_inc  = sidetomove(board) == WHITE ? winc  : binc
            if my_time > 0
                time_limit = clamp(div(my_time, 20) + div(my_inc, 2), 0, div(my_time, 2))
            end
        end
    end

    Threads.@spawn search_thread(deepcopy(board), search_depth, time_limit)
end

function process_stop()
    search_running[] && (search_stopped[] = true)
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
    if DEV_MODE
        println("option name RFP_Margin type spin default 175 min 0 max 1000")
        println("option name RFP_Improving type spin default 25 min 0 max 500")
        println("option name RFP_MaxDepth type spin default 8 min 0 max 32")
        println("option name NMP_Base type spin default 4 min 1 max 16")
        println("option name NMP_DepthDiv type spin default 3 min 1 max 16")
        println("option name NMP_EvalDiv type spin default 200 min 10 max 2000")
        println("option name NMP_MaxExtra type spin default 3 min 0 max 16")
        println("option name NMP_MinDepth type spin default 3 min 1 max 32")
        println("option name LMP_Base type spin default 5 min 0 max 50")
        println("option name LMP_Div type spin default 2 min 1 max 10")
        println("option name LMP_MaxDepth type spin default 8 min 0 max 32")
        println("option name LMR_Divisor type spin default 3 min 1 max 20")
        println("option name LMR_InCheck type spin default -1 min -10 max 10")
        println("option name LMR_Capture type spin default -1 min -10 max 10")
        println("option name LMR_NonPV type spin default 1 min 0 max 10")
        println("option name Asp_Window type spin default 50 min 1 max 1000")
        println("option name Asp_Depth type spin default 6 min 1 max 32")
        println("option name Asp_Max type spin default 200 min 10 max 10000")
        println("option name IIR_MinDepth type spin default 3 min 1 max 32")
        println("option name SE_Margin type spin default 6 min 0 max 100")
        println("option name SE_MinDepth type spin default 6 min 1 max 32")
        println("option name SE_DoubleMargin type spin default 40 min 0 max 500")
        println("option name MaxHistory type spin default 16384 min 256 max 65536")
        println("option name Corr_Delta type spin default 256 min 1 max 65536")
        println("option name Corr_Learning type spin default 64 min 1 max 65536")
        println("option name Corr_Gamma type spin default 16384 min 1 max 65536")
    end
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
