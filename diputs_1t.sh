#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia --threads=1 --project="$DIR" "$DIR/src/uci.jl"
