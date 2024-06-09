XLEN ?= 32

TOOLDIR ?= /opt

TARGET ?= opaesim

XRT_SYN_DIR ?= ../../../hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
VX_CFLAGS += -march=rv64imafd -mabi=lp64d
STARTUP_ADDR ?= 0x180000000
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
STARTUP_ADDR ?= 0x80000000
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

VORTEX_RT_PATH ?= $(realpath ../../../runtime)
VORTEX_KN_PATH ?= $(realpath ../../../kernel)
GEMMINI_SW_PATH ?= $(realpath ../../../third_party/gemmini-rocc-tests)

FPGA_BIN_DIR ?= $(VORTEX_RT_PATH)/opae

LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
#LLVM_CFLAGS += -mllvm -vortex-branch-divergence=2
#LLVM_CFLAGS += -mllvm -print-after-all 
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0/$(RISCV_PREFIX) 
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/$(RISCV_PREFIX)/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump 
VX_CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

#VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
#VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-g++
#VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
#VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

VX_CFLAGS += -v -O3 -std=c++17
VX_CFLAGS += -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections -mllvm -inline-threshold=8192
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw -I$(GEMMINI_SW_PATH)
VX_CFLAGS += -DNDEBUG -DLLVM_VORTEX

# VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortexrt.a
VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortexrt.a $(VORTEX_KN_PATH)/tohost.S

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_KN_PATH)/../hw

LDFLAGS += -L$(VORTEX_RT_PATH)/stub -lvortex

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif	
endif
endif

# CONFIG is supplied from the command line to differentiate ELF files with custom suffixes
CONFIGEXT = $(if $(CONFIG),.$(CONFIG),)

all: $(PROJECT) kernel.bin kernel.dump kernel.radiance.dump kernel$(CONFIGEXT).dump kernel.radiance$(CONFIGEXT).dump

kernel.dump: kernel.elf
	$(VX_DP) -D kernel.elf > kernel.dump

kernel.radiance.dump: kernel.radiance.elf
	$(VX_DP) -D kernel.radiance.elf > kernel.radiance.dump

ifneq ($(CONFIG),)
kernel$(CONFIGEXT).dump: kernel$(CONFIGEXT).elf
	$(VX_DP) -D kernel$(CONFIGEXT).elf > kernel$(CONFIGEXT).dump

kernel.radiance$(CONFIGEXT).dump: kernel.radiance$(CONFIGEXT).elf
	$(VX_DP) -D kernel.radiance$(CONFIGEXT).elf > kernel.radiance$(CONFIGEXT).dump
endif

kernel.bin: kernel.elf kernel.radiance.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

OBJCOPY ?= $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy
OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
kernel.elf: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o $@
	$(OBJCOPY) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --update-section .operand.a=input.a.bin $@
	$(OBJCOPY) --update-section .operand.b=input.b.bin $@
	$(OBJCOPY) --update-section .args=args.bin $@

kernel.radiance.elf: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -DRADIANCE -o $@
	$(OBJCOPY) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --update-section .operand.a=input.a.bin $@
	$(OBJCOPY) --update-section .operand.b=input.b.bin $@
	$(OBJCOPY) --update-section .args=args.bin $@

ifneq ($(CONFIG),)
kernel$(CONFIGEXT).elf: kernel.elf
	cp $< $@

kernel.radiance$(CONFIGEXT).elf: kernel.radiance.elf
	cp $< $@
endif

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-opae: $(PROJECT) kernel.bin   
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(VORTEX_RT_PATH)/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.bin
ifeq ($(TARGET), hw)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
else
	XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)	
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf kernel.elf kernel.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
