#!/bin/sh

# Copyright 2023 blaise
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TOOLDIR=${TOOLDIR:=$HOME/build/vortex-toolchain-prebuilt}
export TOOLDIR

export VERILATOR_ROOT=$TOOLDIR/verilator
export PATH=$VERILATOR_ROOT/bin:$PATH

export SV2V_PATH=$TOOLDIR/sv2v
export PATH=$SV2V_PATH/bin:$PATH

export YOSYS_PATH=$TOOLDIR/yosys
export PATH=$YOSYS_PATH/bin:$PATH

# LLVM_POCL seems to be only used in tests/opencl
export LLVM_POCL=/home/virgo-ae/build/llvm-vortex2
export LLVM_VORTEX=/home/virgo-ae/build/llvm-vortex2
export POCL_CC_PATH=/home/virgo-ae/build/pocl-vortex2/compiler
export POCL_RT_PATH=/home/virgo-ae/build/pocl-vortex2/runtime
