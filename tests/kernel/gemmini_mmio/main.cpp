#include <stdio.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <include/gemmini.h>
#include "gemmini_mmio.h"

int main() {

  char *print_buf = (char *) PRINT_BUF;

  sprintf(print_buf, "\n%d\n", DIM);

  gemmini_config_ld(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);

  // load up A and B and C
  uint32_t spad_A = 0x00000000;
  uint32_t spad_B = 0x00000100; // 16B word addressed
  uint32_t acc_C = 0x80000000; // accmem + accumulate
  uint32_t spad_C = 0x00000200;

  float *smem_A = (float *) SPAD_TO_SMEM(spad_A); // 0xff000000; // byte addressed
  float *smem_B = (float *) SPAD_TO_SMEM(spad_B); // 0xff000200;
  float *smem_C = (float *) SPAD_TO_SMEM(spad_C); // 0xff000400;

  int I = 5;
  int J = 5;
  int K = 5;

  gemmini_config_st(DIM * 4 * J)

  // load A with 128->1 in row-major order
  for (int i = 0; i < I; i++) {
    for (int k = 0; k < K; k++) {
      int tile_byte_offset = (i * K + k) * DIM * DIM;
      for (int x = 0; x < DIM; x++)
        for (int y = 0; y < DIM; y++)
          smem_A[tile_byte_offset + x * DIM + y] = (float) ((I * K * DIM * DIM - ((i * DIM + x) * DIM * K + (k * DIM + y))) % 64);
    }
  }

  // load B with 0->191 in row-major order
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < J; j++) {
      int tile_byte_offset = (k * J + j) * DIM * DIM;
      for (int x = 0; x < DIM; x++)
        for (int y = 0; y < DIM; y++)
          smem_B[tile_byte_offset + x * DIM + y] = (float) (((k * DIM + x) * DIM * J + (j * DIM + y)) % 64);
    }
  }

  for (int i = 0; i < I * J * DIM * DIM; i++) smem_C[i] = 0.f;

  pfence();

  // sprintf(print_buf, "\nA in\n");
  // for (int i = I * DIM - 1; i < I * DIM; i++) {
  //   for (int j = 0; j < K * DIM; j++) {
  //     sprintf(print_buf, "%d ", (int) (smem_A[SMEM_MAT_OFFSET(i, j, K * DIM)]));
  //   }
  //   sprintf(print_buf, "\n");
  // }

  // sprintf(print_buf, "\nB in\n");
  // for (int i = 0; i < K * DIM; i++) {
  //   for (int j = 0; j < J * DIM; j++) {
  //     sprintf(print_buf, "%d ", (int) (smem_B[SMEM_MAT_OFFSET(i, j, J * DIM)]));
  //   }
  //   sprintf(print_buf, "\n");
  //   if (i == 2) i = K * DIM - 3;
  // }

  // gemmini_extended_preload(spad_B, acc_C, DIM, DIM, DIM, DIM);
  // gemmini_extended_compute_preloaded(spad_A, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
  // gemmini_extended_mvout(0xc0000000, 0xff000000, DIM, DIM);
  // gemmini_extended_mvout_spad(spad_C, 1, acc_C, DIM, DIM);
  
  sp_tiled_matmul_full_spad_ws(spad_A, spad_B, /*spad_D=*/0, spad_C,
      /*I=*/I, /*J=*/J, /*K=*/K, /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
      /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
      /*no_bias=*/1, /*repeating_bias=*/0, /*act=*/NO_ACTIVATION);

  for (int i = 0; i < 32; i++) pfence();

  // check results
  for (int i = 0; i < I * DIM; i++) {
    for (int j = 0; j < J * DIM; j++) {
      int sum = 0;
      for (int k = 0; k < K * DIM; k++) sum += ((I * K * DIM * DIM - i * K * DIM - k) % 64) * ((k * J * DIM + j) % 64);
      if ((int) (smem_C[SMEM_MAT_OFFSET(i, j, J * DIM)] * 10) != (int) (sum * 10)) {
        sprintf(print_buf, "TEST FAILED (actual/reference)\n");
        for (int ii = 0; ii < I * DIM; ii++) {
          for (int jj = 0; jj < J * DIM; jj++) {
            sum = 0;
            for (int k = 0; k < K * DIM; k++) sum += ((I * K * DIM * DIM - ii * K * DIM - k) % 64) * ((k * J * DIM + jj) % 64);
            sprintf(print_buf, "%d/%d ", (int) (smem_C[SMEM_MAT_OFFSET(ii, jj, J * DIM)]), (int) sum);
          }
          sprintf(print_buf, "\n");
        }
        return 1;
      }
    }
  }
  sprintf(print_buf, "TEST PASSED\n");

  return 0;
}
