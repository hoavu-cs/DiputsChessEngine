# Diputs Chess Engine

A non-competitive UCI chess engine for trolling.
I have ~~1 week~~ 2 weeks to work on this engine to brush up on Julia.
Julia, like Python, is highly portable and easy on the eyes.

**Strength**. This engine plays chess with an Elo. 
60 games between commit a1e499a against Stash 37, 2:00+1, UHO_Lichess_4852_v1.epd, 1 thread, 256Mb hash each.
```
Score of diputs.sh vs stash-37.0-linux-64: 34 - 11 - 15 [0.692]
```

**Running using a Wrapper Script and GUI**. Requirements: [Julia](https://julialang.org/downloads/) 1.9+. Make sure to add Julia to your PATH (while it's possible to create a binary using [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/), it kind of defeats the purpose of a fast scripting language).

Point your UCI-compatible GUI (Arena, Cutechess, En Croissant, Nibbler etc.) to the wrapper script:

```bash
diputs_1t.sh   # 1-thread version
diputs_2t.sh   # 2-thread version
diputs_4t.sh   # 4-thread version
```

Make them executable first:

Then point the GUI program to it or run in the terminal.

```bash
./diputs_1t.sh
uci
isready
position startpos
go depth 20
...
quit
```


**Running using Julia Directly**. You can also run the engine directly from the terminal.

```bash
julia --project=. src/uci.jl 
```

**SMP (multi-threaded).** The thread count is fixed at Julia startup, to run with 1, 2, or 4 threads, use: `diputs_1t.sh`, `diputs_2t.sh`, or `diputs_4t.sh` respectively. You can also create your own wrapper script with a custom thread count:

```bash
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia --threads=16 --project="$DIR" "$DIR/src/uci.jl"
```

make it executable, and then point your GUI to it. 2-threaded version gains around 50 Elo against the single-threaded version (based on testing). For normal use, 2 threads are recommended. I haven't tested higher thread count thoroughly.


### License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
