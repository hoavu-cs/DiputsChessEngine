# Diputs Chess Engine

A non-competitive troll UCI chess engine written in Julia with some vibecoding.
I have ~~1 week~~ 2 weeks to work on this project (mostly to rehash my Julia) and then I will stop (most likely).
Julia, like Python, is highly portable and easy on the eye.

## Strength
Commit 82bfebb against Stash 27 (~3050), 3+2. UHO_Lichess_4852_v1.epd.
```
Score of diputs.sh vs stash-27.0-linux-64: 63 - 17 - 26 [0.717]
...      diputs.sh playing White: 38 - 7 - 10  [0.782] 55
...      diputs.sh playing Black: 25 - 10 - 16  [0.647] 51
...      White vs Black: 48 - 32 - 26  [0.575] 106
Elo difference: 161.5 +/- 62.3, LOS: 100.0 %, DrawRatio: 24.5 %
SPRT: llr 0 (0.0%), lbound -inf, ubound inf

```

## Requirements

- [Julia](https://julialang.org/downloads/) 1.9+. Make sure to add it to your PATH.

## Setup (one time)

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

## Running using Julia Directly

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

## Adding to a GUI using Wrapper Script

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
LOS: 100.00 %, DrawRatio: 41.11 %, PairsRatio: 2.03
Games: 360, Wins: 127, Losses: 74, Draws: 159, Points: 206.5 (57.36 %)
Ptnml(0-2): [1, 34, 74, 53, 18], WL/DD Ratio: 1.06
LLR: 2.24 (101.8%) (-2.20, 2.20) [0.00, 10.00]
```
