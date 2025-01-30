#!/bin/sh

echo "generating operands"

python3 generate_operands.py

for a in args/*; do
    echo "compiling GEMM kernel for Virgo with dim ${a}"
    cp -f $a args.bin
    aa=$(basename "$a")
    cp -f input.a.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.a.bin
    cp -f input.b.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.b.bin

    # touch source file to force re-building, as the Makefile does not track
    # binary changes
    touch kernel.cpp

    make CONFIG=gemm.virgo.hopper.dim${aa}
done
