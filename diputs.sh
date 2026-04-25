#!/bin/bash
DIR=/home/hoa-vu/git/diputs-chess-engine
exec julia --project="$DIR" "$DIR/uci.jl"
