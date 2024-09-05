PROJECT = tensor

SRCS = main.cpp
DEPS = a_matrix.h
DEPS += b_matrix.h
DEPS += c_matrix.h

include ../common.mk

OBJCOPY ?= $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy
OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
BINFILES :=  args.bin input.a.bin input.b.bin
$(PROJECT).elf: $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT).elf
	$(OBJCOPY) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.c=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --update-section .operand.a=input.a.bin $@ || true
	$(OBJCOPY) --update-section .operand.b=input.b.bin $@ || true
	$(OBJCOPY) --update-section .operand.c=input.c.bin $@ || true
	$(OBJCOPY) --update-section .args=args.bin $@ || true
