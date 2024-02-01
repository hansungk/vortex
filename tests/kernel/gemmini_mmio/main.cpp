#include <stdio.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <include/gemmini.h>

// #define ADDR_LEN 32
// #define XCUSTOM_ACC 3
// #define k_MVOUT_SPAD 23

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    /* printf("function %d\n", funct); */ \
    uint32_t instruction = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((uint32_t) funct << 25); \
    *((volatile uint64_t*) 0xff002010) = (uint64_t) (rs1); \
    *((volatile uint64_t*) 0xff002018) = (uint64_t) (rs2); \
    /* gemmini_fence(); */ \
    *((volatile uint32_t*) 0xff002000) = instruction; \
}

// #define gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows) \
//   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(dst_stride) << 32) | (uint64_t)(dst_addr), ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(src_addr), k_MVOUT_SPAD)

// #define gemmini_mvout_spad(dst_addr, src_addr, cols, rows) \
//   gemmini_extended_mvout_spad(dst_addr, 1, src_addr, cols, rows)

int main() {
  volatile uint64_t *bogus = (uint64_t *)0x00001000;

  gemmini_config_ld(0);
  gemmini_config_st(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);

  // bogus loop to give slack for MMIO to settle without fences
  for (int i = 0; i < 10; i++) {
    *bogus = 0xdeadbeef;
  }

  // load up A and B and C
  float *A = (float *)0xff000000;
  float *B = (float *)0xff000100;
  float *C = (float *)0xff000200;
  float *D = (float *)0xff000300;
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      A[i * DIM + j] = 1.0f;
      B[i * DIM + j] = 1.0f;
      C[i * DIM + j] = 0.0f;
      D[i * DIM + j] = 0.0f;
    }
  }

  for (int i = 0; i < 10; i++) {
    *bogus = 0xdeadbeef;
  }

  gemmini_extended_preload(B, C, DIM, DIM, DIM, DIM);

  for (int i = 0; i < 10; i++) {
    *bogus = 0xdeadbeef;
  }

  gemmini_extended_compute_preloaded(A, D, DIM, DIM, DIM, DIM);

  for (int i = 0; i < 10; i++) {
    *bogus = 0xdeadbeef;
  }

  // gemmini_extended_mvout(0xc0000000, 0xff000000, DIM, DIM);
  gemmini_mvout_spad(0x00000000, 0x00000200/*C*/, DIM, DIM);

  for (int i = 0; i < 100; i++) {
    *bogus = 0xdeadbeef;
  }

  return 0;
}
