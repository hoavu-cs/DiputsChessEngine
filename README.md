# Diputs Chess Engine

A non-competitive UCI chess engine for trolling written in Julia.
I have ~~1 week~~ 2 weeks to work on this project to rehash Julia.
Julia, like Python, is highly portable and easy on the eye.

Requirements: [Julia](https://julialang.org/downloads/) 1.9+. Make sure to add it to your PATH.

## Strength
This engine plays chess with an Elo.
100 games between commit 09ee592 against Stash 30 (~3164), 2:00+1, UHO_Lichess_4852_v1.epd.

```
Score of diputs.sh vs stash-30.0-linux-64: 49 - 24 - 27 [0.625]
Elo difference: 88.7 +/- 60.0, LOS: 99.8 %, DrawRatio: 27.0 %
```

## Setup (one time)

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

## Running using Julia Directly
You can run the engine directly from the terminal.

```bash
julia --project=. src/uci.jl
```

Communicates via the UCI protocol on stdin/stdout. Quick smoke test:

```bash
julia --project=. src/uci.jl 
uci
isready
position startpos
go depth 20
...
quit
```

## Running using a Wrapper Script and GUI

Point your UCI-compatible GUI (Arena, Cutechess, En Croissant, Nibbler etc.) to the wrapper script. For single-threaded version:
```
diputs_1t.sh
```

Make it executable first:

```bash
chmod +x diputs_1t.sh
```

Or run directly from the terminal:

```bash
bash diputs_1t.sh
```


## SMP (multi-threaded)

The thread count is fixed at Julia startup, to run with 1, 2, 4, or 8 threads, use: `diputs_1t.sh`, `diputs_2t.sh`, `diputs_4t.sh`, or `diputs_8t.sh` respectively. You can also create your own wrapper script with a custom thread count. Say you can create `diputs_16t.sh` with 16 threads like this:
```bash
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia --threads=16 --project="$DIR" "$DIR/src/uci.jl"
```

make it executable, and then point your GUI to it. 

2-threaded version gains around 50 Elo against the single-threaded version (based on testing). It is known that [lazy SMP](https://www.chessprogramming.org/Lazy_SMP) scales up to 8 cores and above.

```
Results of DIPUTEXP-SMP vs Diputs (10+0.1, NULL - 4t, 256MB, UHO_Lichess_4852_v1.epd):
Elo: 51.52 +/- 23.51, nElo: 79.94 +/- 35.89
```



## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
