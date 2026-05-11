#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia --threads=2 --project="$DIR" "$DIR/src/uci.jl"
