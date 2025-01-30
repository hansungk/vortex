rm kernel.radiance.elf
rm -rf binaries
mkdir binaries
for a in args/*; do
    cp -f $a args.bin
    aa=$(basename "$a")
    cp -f input.a/"$aa" input.a.bin
    cp -f input.b/"$aa" input.b.bin
    make > /dev/null
    mv kernel.radiance.elf binaries/gemmini_fp16nodma"$aa".elf
done
