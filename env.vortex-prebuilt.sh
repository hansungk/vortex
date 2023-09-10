if [ -n "$VORTEX_ENV" ]
then
    echo "VORTEX_ENV already set. Exiting."
    return
fi

# PREBUILT_DIR=/scratch/hansung/build/vortex-toolchain-prebuilt-d2ba5df-230831
PREBUILT_DIR=/scratch/hansung/build/vortex-toolchain-prebuilt-230831

export VORTEX_ENV="vortex-prebuilt"
export LLVM_PREFIX=$PREBUILT_DIR/llvm-riscv/
export POCL_CC_PATH=$PREBUILT_DIR/pocl/compiler
export POCL_RT_PATH=$PREBUILT_DIR/pocl/runtime
export VERILATOR_ROOT=$PREBUILT_DIR/verilator
export RISCV_TOOLCHAIN_PATH=$PREBUILT_DIR/
export PATH="$BUILDDIR/vortex-toolchain-prebuilt-d2ba5df-230831/verilator/bin:$PATH"
export PS1="($VORTEX_ENV) $PS1"
