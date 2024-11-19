Example Headless Simulation
===========================

This directory contains an example kernel for a headless simulation of the
Vortex GPGPU, which eliminates involvement of the host-side CPU execution and
enables stand-alone simulation of the GPU.

## Overview

There are three binaries needed for running a kernel on Vortex:

* **Program binary**: Vortex ISA program binary, situated at `0x80000000` in
  the device memory.
* **Argument binary**: Binary that contains the kernel argument.  This is normally
  uploaded from host to the device memory, using `vx_copy_to_dev()` in the host
  program (`main.cpp`).  It is located at a statically known address defined as
  `KERNEL_ARG_DEV_MEM_ADDR` in [`common.h`](common.h).
* **Input binary**: Binary that contains auxiliary input data to the kernel,
  e.g. vector data of FP32 elements used as input in a vector-add kernel. Its
  memory address is communicated to the kernel through the kernel argument,
  which contains a pointer field. See `kernel_arg_t.addr_{a,b,dst}` in
  [`common.h`](common.h).

The idea is to load all of these binaries to the device memory altogether by
_stitching them into a single program binary._  In order to do this, we modify
the [`vx_link32.ld` linker script](/kernel/linker/vx_link32.ld) to add new
sections for the argument and input binaries in the ELF file:

```
...
MEMORY {
  DRAM0    (rwx): ORIGIN = 0x80000000, LENGTH = 512M
  DRAMARG  (rwx): ORIGIN = 0x9fff0000, LENGTH = 8K
  DRAM1    (rwx): ORIGIN = 0xa0000000, LENGTH = 16M
  DRAM2    (rwx): ORIGIN = 0xa1000000, LENGTH = 16M
  DRAM3    (rwx): ORIGIN = 0xa2000000, LENGTH = 16M
}
...
  .args : {
    *(.args)
    . += 8K;
  }> DRAMARG
  .operand.a : {
    *(.operand.a)
    . += 32K;
  }> DRAM1
  .operand.b : {
    *(.operand.b)
    . += 32K;
  }> DRAM2
  .operand.c : {
    *(.operand.c)
    . += 32K;
  }> DRAM3
...
```

Then, when compiling the kernel, we use the `objcopy` tool to stitch the bits
from separate binary files into the corresponding new sections in `kernel.elf`
using the `--update-section` flag in [`common.mk`](/tests/regression/common.mk):

```
...
kernel.elf: $(VX_SRCS) $(VX_INCLUDES) $(BINFILES)
	$(VX_CXX) $(VX_CFLAGS) -o $@ $(VX_SRCS) $(VX_LDFLAGS)
	$(OBJCOPY) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.c=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --update-section .operand.a=input.a.bin $@ || true
	$(OBJCOPY) --update-section .operand.b=input.b.bin $@ || true
	$(OBJCOPY) --update-section .operand.c=input.c.bin $@ || true
	$(OBJCOPY) --update-section .args=args.bin $@ || true
...
```
