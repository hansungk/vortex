XLEN ?= 32
TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
CFLAGS += -march=rv64imafd -mabi=lp64d
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
CFLAGS += -march=rv32imaf -mabi=ilp32f
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf

VORTEX_KN_PATH ?= $(realpath ../../../kernel)

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

SIM_DIR = ../../../sim

CFLAGS += -O3 -mcmodel=medany -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -ffixed-ft0 -ffixed-ft1 -ffixed-ft2 -ffixed-ft3 -ffixed-ft4 -ffixed-ft5 -ffixed-ft6 -ffixed-ft7
CFLAGS += -ffixed-fs0 -ffixed-fs1 -ffixed-fs2 -ffixed-fs3 -ffixed-fs4 -ffixed-fs5 -ffixed-fs6 -ffixed-fs7
CFLAGS += -ffixed-fa0 -ffixed-fa1 -ffixed-fa2 -ffixed-fa3 -ffixed-fa4 -ffixed-fa5 -ffixed-fa6 -ffixed-fa7
CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw

LDFLAGS += -lm -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=0x80000000 $(VORTEX_KN_PATH)/libvortexrt.a

all: $(PROJECT).elf $(PROJECT).bin $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $(PROJECT).elf > $(PROJECT).dump

$(PROJECT).bin: $(PROJECT).elf
	$(CP) -O binary $(PROJECT).elf $(PROJECT).bin

$(PROJECT).elf: $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT).elf

run-rtlsim: $(PROJECT).bin
	$(SIM_DIR)/rtlsim/rtlsim $(PROJECT).bin

run-simx: $(PROJECT).bin
	$(SIM_DIR)/simx/simx $(PROJECT).bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump .depend 
