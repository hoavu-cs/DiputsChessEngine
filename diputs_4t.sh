#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia --threads=5 --project="$DIR" "$DIR/src/uci.jl"
