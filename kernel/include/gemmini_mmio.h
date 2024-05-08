#ifndef GEMMINI_MMIO_H
#define GEMMINI_MMIO_H
#ifndef GEMMINI_PARAMS_H
#error INCLUDE GEMMINI.H FIRST
#endif

#define SMEM_BASE 0xff000000
#define SMEM_SIZE 0x4000
#define SMEM_MASK (SMEM_SIZE - 1)
#define SMEM_ADDR_END 0xff008000

#define SPAD_BASE 0x0
#define SPAD_ROW_SIZE (DIM * sizeof(elem_t))
#define SPAD_NUM_ROWS (SMEM_SIZE / SPAD_ROW_SIZE)
#define SPAD_MASK (SPAD_NUM_ROWS - 1)

#define PRINT_BUF ((char *) (SMEM_ADDR_END))
#define GEMMINI_RS1_ADDR 0xff007010
#define GEMMINI_RS2_ADDR 0xff007018
#define GEMMINI_INST_ADDR 0xff007000
#define GEMMINI_BUSY_ADDR 0xff007020

#define SMEM_TO_SPAD(smem_addr) (SPAD_BASE + ((smem_addr) & SMEM_MASK) / SPAD_ROW_SIZE)
#define SPAD_TO_SMEM(spad_addr) (SMEM_BASE + ((spad_addr) & SPAD_MASK) * SPAD_ROW_SIZE)

// convert normal matrix i,j into tiled smem offset
// top_in_tiles = i / DIM
// left_in_tiles = j / DIM
// num_tiles_before_current = top_in_tiles * (J / DIM) + left_in_tiles
// smem_addr = num_tiles_before_current * DIM * DIM + (i % DIM) * DIM + (j % DIM)
#define SMEM_MAT_OFFSET(i, j, J) \
    (((i) / DIM * (J) / DIM + (j) / DIM) * DIM * DIM + ((i) % DIM) * DIM + ((j) % DIM))

// #define fence() { for (int i = 0; i < 10; i++) *((volatile uint32_t *) (0xFFFF0000)) = 0xdeadbeef; }
#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    /* printf("function %d\n", funct); */              \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    /* *((volatile uint32_t*) GEMMINI_RS2_ADDR) = (uint32_t) ((uint64_t) (rs2) & 0xFFFFFFFFULL); */ \
    /* *((volatile uint32_t*) (GEMMINI_RS2_ADDR + 4)) = (uint32_t) ((uint64_t) (rs2) >> 32); */ \
    /* gemmini_fence(); */ \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
    /* sprintf((char *) PRINT_BUF, "%llx %llx %d\n", rs1, rs2, funct); */ \
}

#define sp_tiled_matmul_full_spad_ws(A_sp_addr_start, B_sp_addr_start, D_sp_addr_start, C_dst_sp_addr_start,\
  I, J, K, pad_I, pad_J, pad_K, a_transpose, b_transpose, full_C, low_D, acc, act, skips) \
  gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A_sp_addr_start, (B_sp_addr_start) + (K) * (J) * DIM, NULL, \
  C_dst_sp_addr_start, a_transpose, b_transpose, full_C, low_D, acc, act, 0, 0, false, skips)

/* inline static void sp_tiled_matmul_full_spad_ws(const uint32_t A_sp_addr_start, const uint32_t B_sp_addr_start,
                                                const uint32_t D_sp_addr_start, const uint32_t C_dst_sp_addr_start,
                                                size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
                                                bool a_transpose, bool b_transpose,
                                                bool full_C, bool low_D, bool acc,
                                                int act, int skip_mvout) {

  gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K,
                       A_sp_addr_start, B_sp_addr_start + K * J * DIM, NULL, C_dst_sp_addr_start,
                       a_transpose, b_transpose,
                       full_C, low_D, acc,
                       act, 0, 0, false, skip_mvout); */
  /*
  return;


  // const uint32_t A_sp_addr_start = 0;
  // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  // const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 2 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  // const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
  //   (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = 1; //full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  // const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
  gemmini_fence();

  if (a_transpose || b_transpose || (I < 4)) {
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < J; j++) {
        for (size_t i = 0; i < I; i++) {
          const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
            (A_sp_addr_start + (i*K + k)*DIM);
          const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
            (B_sp_addr_start + (k*J + j)*DIM);
          const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
          // Compute
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr | ((k == 0 ? 0 : 1) << (ADDR_LEN-2));
          gemmini_extended_preload(pre_sp_addr, out_sp_addr, DIM, DIM, DIM, DIM);
          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          }
          if (k == K - 1) {
            // Move-out C (if not normalizing)
            // if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
              const size_t rounded_j = j; // (j / C_blocks) * C_blocks;
              const uint32_t rounded_C_sp_addr = C_sp_addr; // C_sp_addr_start + (i*J + rounded_j)*DIM;

              const uint32_t C_dst_sp_addr = ((uint32_t) C_dst_sp_addr_start) + (i * J + rounded_j) * DIM; // * DIM * sizeof_C;

              // const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
              constexpr size_t cols = DIM; // blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
              constexpr size_t rows = DIM; // DIM - (i == I - 1 ? pad_I : 0);

              gemmini_extended_mvout_spad(C_dst_sp_addr, 1, rounded_C_sp_addr, cols, rows);
            // }
          }
        }
      }
    }
  } else {
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < J; j++) {
        uint32_t A_sp_addr = A_sp_addr_start + k * DIM; // (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
        uint32_t C_sp_addr = C_sp_addr_start + j * DIM; // (i*J + j)*DIM;
        for (size_t i = 0; i < I; i += 4) {
          // Compute
          // constexpr uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          const uint32_t out_sp_addr = C_sp_addr | ((k == 0 ? 0 : 1) << (ADDR_LEN-2));
          if (i == 0) { // First iteration
            gemmini_extended_preload(B_sp_addr, out_sp_addr, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 2 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 2 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 3 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 3 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 2 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 2 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 3 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 3 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          }
          if (k == K - 1) {
            for (int x = 0; x < 3; x++) gemmini_fence();
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + (i * J + j) * DIM, 1, C_sp_addr, DIM, DIM);
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + ((i + 1) * J + j) * DIM, 1, C_sp_addr + J * DIM, DIM, DIM);
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + ((i + 2) * J + j) * DIM, 1, C_sp_addr + 2 * J * DIM, DIM, DIM);
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + ((i + 3) * J + j) * DIM, 1, C_sp_addr + 3 * J * DIM, DIM, DIM);
          }
          A_sp_addr += 4 * K * DIM;
          C_sp_addr += 4 * J * DIM;
        }
      }
    }
  }
  gemmini_fence();
}*/


#endif
