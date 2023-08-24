if [ $# -lt 1 ]
then
    echo "Usage: source env.vortex-prebuilt.sh build-dir"
    return
fi

if [ -n "$VORTEX_ENV" ]
then
    echo "VORTEX_ENV already set. Exiting."
    return
fi

BUILDDIR=$1

if ! [ -d "$BUILDDIR/vortex-toolchain-prebuilt" ]
then
    echo "error: $BUILDDIR/vortex-toolchain-prebuilt does not exist."
    return 1
fi

export VORTEX_ENV="vortex-prebuilt"
export LLVM_PREFIX=$BUILDDIR/vortex-toolchain-prebuilt/llvm-riscv/
export POCL_CC_PATH=$BUILDDIR/vortex-toolchain-prebuilt/pocl/compiler
export POCL_RT_PATH=$BUILDDIR/vortex-toolchain-prebuilt/pocl/runtime
export VERILATOR_ROOT=$BUILDDIR/vortex-toolchain-prebuilt/verilator
export RISCV_TOOLCHAIN_PATH=$BUILDDIR/vortex-toolchain-prebuilt
export PATH="$BUILDDIR/vortex-toolchain-prebuilt/verilator/bin:$PATH"
export PS1="($VORTEX_ENV) $PS1"
