DESTDIR ?= .
RTL_DIR = ../../rtl
DPI_DIR = $(abspath ../../dpi)
SIM_DIR = ../../../sim
THIRD_PARTY_DIR = $(abspath ../../../third_party)

CONFIGS +=
PARAMS +=

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -fcoroutines
CXXFLAGS += -I../../.. -I../../common -I../../../../sim/common
CXXFLAGS += -I/$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I/$(DPI_DIR)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_BANK 
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_MSHR
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_TAG
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_DATA

DBG_FLAGS += -DDEBUG_LEVEL=$(DEBUG) -DVCD_OUTPUT $(DBG_TRACE_FLAGS)

RTL_PKGS = $(RTL_DIR)/VX_gpu_pkg.sv

RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/fpu

# SRCS = cachesim.cpp testbench.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp
SRCS += $(DPI_DIR)/float_dpi.cpp
SRCS += $(SIM_DIR)/common/rvfloats.cpp
SRCS += ./main.cpp

RTL_SRCS += $(RTL_DIR)/fpu/VX_tensor_core.sv
RTL_SRCS += $(RTL_DIR)/fpu/VX_tensor_tb.sv

TOP = VX_tensor_tb

VL_FLAGS = --exe
VL_FLAGS += --language 1800-2009 # -Wall -Wpedantic # --assert 
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += -DSIMULATION -DSV_DPI
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(PARAMS)
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(RTL_PKGS)
VL_FLAGS += --cc $(TOP) --top-module $(TOP)
VL_FLAGS += --timing

# Enable Verilator multithreaded simulation
THREADS ?= $(shell python -c 'import multiprocessing as mp; print(mp.cpu_count())')
VL_FLAGS += -j $(THREADS)
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace --trace-structs $(DBG_FLAGS)
	CXXFLAGS += -g -O0 $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	VL_FLAGS += -DPERF_ENABLE
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = tensor

all: $(DESTDIR)/$(PROJECT)
	
$(DESTDIR)/$(PROJECT): $(SRCS) $(RTL_SRCS)
	verilator --build $(VL_FLAGS) $(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$@

run: $(DESTDIR)/$(PROJECT)
	$(DESTDIR)/$(PROJECT)

waves: trace.vcd
	gtkwave -o trace.vcd

clean:
	rm -rf obj_dir $(DESTDIR)/$(PROJECT)
