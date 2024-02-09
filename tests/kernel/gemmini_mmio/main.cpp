#include <stdio.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <include/gemmini.h>

// #define ADDR_LEN 32
// #define XCUSTOM_ACC 3
// #define k_MVOUT_SPAD 23

#define pfence() { for (int i = 0; i < 10; i++) *((uint32_t *) 0xffff0000) = 0xdeadbeef; }

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    /* printf("function %d\n", funct); */ \
    uint32_t instruction = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((uint32_t) funct << 25); \
    *((volatile uint64_t*) 0xff100010) = (uint64_t) (rs1); \
    *((volatile uint64_t*) 0xff100018) = (uint64_t) (rs2); \
    pfence(); \
    /* gemmini_fence(); */ \
    *((volatile uint32_t*) 0xff100000) = instruction; \
}

// #define gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows) \
//   ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(dst_stride) << 32) | (uint64_t)(dst_addr), ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(src_addr), k_MVOUT_SPAD)

// #define gemmini_mvout_spad(dst_addr, src_addr, cols, rows) \
//   gemmini_extended_mvout_spad(dst_addr, 1, src_addr, cols, rows)

int main() {

  char *print_buf = ((char *) 0xff005000);
  sprintf(print_buf, "hello world\n");

  gemmini_config_ld(0);
  gemmini_config_st(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);

  // bogus loop to give slack for MMIO to settle without fences

  // load up A and B and C
  float *smem_A = (float *)0xff000000; // byte addressed
  uint32_t spad_A = 0x00000000;
  float *smem_B = (float *)0xff000040;
  uint32_t spad_B = 0x00000004; // 16B word addressed
  float *smem_C = (float *)0xff000080;
  uint32_t acc_C = 0x80000000;
  uint32_t spad_C = 0x00000008;
  float *smem_D = (float *)0xff0000c0;
  uint32_t spad_D = 0x0000000c;

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      smem_A[i * DIM + j] = 1.0f;
      smem_B[i * DIM + j] = 1.0f;
      smem_C[i * DIM + j] = 0.0f;
      smem_D[i * DIM + j] = 0.0f;
    }
  }
  pfence();
  sprintf(print_buf, "\nC before\n");
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      sprintf(print_buf, "%d ", (int) (smem_C[i * DIM + j]));
    }
    sprintf(print_buf, "\n");
  }

  pfence();

  gemmini_extended_preload(spad_B, acc_C, DIM, DIM, DIM, DIM);

  pfence();

  gemmini_extended_compute_preloaded(spad_A, spad_D, DIM, DIM, DIM, DIM);

  pfence();

  // gemmini_extended_mvout(0xc0000000, 0xff000000, DIM, DIM);
  gemmini_mvout_spad(spad_C, acc_C, DIM, DIM);

  pfence();


  sprintf(print_buf, "\nC after\n");
  
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      sprintf(print_buf, "%d ", (int) (100 * smem_C[i * DIM + j]));
    }
    sprintf(print_buf, "\n");
  }

  return 0;
}
