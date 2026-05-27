# Diputs Chess Engine

A somewhat uninovative non-competitive UCI chess engine for trolling (same old search techniques and heuristics). I have ~~1 week 2 weeks~~ to work on this engine to brush up on Julia and to correct some of the bugs in my previous C++ engine. Julia, like Python, is highly portable and easy on the eyes (this is probably the only thing interesting about this engine). At the moment, I've used up my time budget; the engine won't be worked on regularly, but I might try a few patches here and there during free time. 

This engine plays chess with an Elo and tries to win at the game of chess.

The current repo's code gives you the strongest version; releases are stable checkpoints.

**Running using a Wrapper Script and GUI**. Requirements: [Julia](https://julialang.org/downloads/) 1.9+. Make sure to add Julia to your PATH (while it's possible to create a binary using [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/), it kind of defeats the purpose of a fast scripting language). 

Point your UCI-compatible GUI (Arena, Cutechess, En Croissant, Nibbler etc.) to the wrapper script:

```bash
diputs_1t.sh   # 1-thread version
diputs_2t.sh   # 2-thread version
```

Make them executable first: `chmod +x diputs_1t.sh`. Then point the GUI program to it or run in the terminal.

```bash
./diputs_1t.sh
uci
isready
position startpos
go 
...
```


**Running using Julia Directly**. You can also run the engine directly using Julia `julia --project=. src/uci.jl`

**SMP (multi-threaded).** The thread count is fixed at Julia startup, to run with 1 or 2, threads, use: `diputs_1t.sh`, or `diputs_2t.sh` respectively. You can also create your own wrapper script with a custom thread count (e.g., `--threads=17`). Allocate 1 extra thread for uci.jl. 2-threaded version has been tested.

### License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
