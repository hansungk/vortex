#include <stdio.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

#define ADDR_LEN 32
#define XCUSTOM_ACC 3
#define k_MVOUT_SPAD 23

// fence
#define gemmini_fence() asm volatile("fence")

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    /* printf("function %d\n", funct); */ \
    uint32_t instruction = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((uint32_t) funct << 25); \
    *((volatile uint64_t*) 0xff002010) = (uint64_t) (rs1); \
    *((volatile uint64_t*) 0xff002018) = (uint64_t) (rs2); \
    /* gemmini_fence(); */ \
    *((volatile uint32_t*) 0xff002000) = instruction; \
}

#define gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(dst_stride) << 32) | (uint64_t)(dst_addr), ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(src_addr), k_MVOUT_SPAD)

#define gemmini_mvout_spad(dst_addr, src_addr, cols, rows) \
  gemmini_extended_mvout_spad(dst_addr, 1, src_addr, cols, rows)

int main() {
  gemmini_mvout_spad(0xff000000, 0xff000100, 4, 4);

  // volatile uint32_t *ptr_cmd = (volatile uint32_t *)0xff100000;
  // *ptr_cmd = 0xdeadbeef;

  return 0;
}
