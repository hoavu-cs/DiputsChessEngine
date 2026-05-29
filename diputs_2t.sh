#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia -O3 --threads=3 --project="$DIR" "$DIR/src/uci.jl"
