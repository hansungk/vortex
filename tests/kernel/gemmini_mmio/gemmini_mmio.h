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

#define SMEM_GARBAGE_ADDR 0xffff0000
#define PRINT_BUF SMEM_ADDR_END
#define GEMMINI_RS1_ADDR 0xff007010
#define GEMMINI_RS2_ADDR 0xff007018
#define GEMMINI_INST_ADDR 0xff007000

#define SMEM_TO_SPAD(smem_addr) (SPAD_BASE + ((smem_addr) & SMEM_MASK) / SPAD_ROW_SIZE)
#define SPAD_TO_SMEM(spad_addr) (SMEM_BASE + ((spad_addr) & SPAD_MASK) * SPAD_ROW_SIZE)

// convert normal matrix i,j into tiled smem offset
// top_in_tiles = i / DIM
// left_in_tiles = j / DIM
// num_tiles_before_current = top_in_tiles * (J / DIM) + left_in_tiles
// smem_addr = num_tiles_before_current * DIM * DIM + (i % DIM) * DIM + (j % DIM)
#define SMEM_MAT_OFFSET(i, j, J) \
    (((i) / DIM * (J) / DIM + (j) / DIM) * DIM * DIM + ((i) % DIM) * DIM + ((j) % DIM))

#define pfence() { for (int i = 0; i < 5; i++) *((volatile uint32_t *) SMEM_GARBAGE_ADDR) = 0xdeadbeef; }

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    /* printf("function %d\n", funct); */ \
    uint32_t instruction = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((uint32_t) (funct) << 25); \
    *((volatile uint64_t*) GEMMINI_RS1_ADDR) = (uint64_t) (rs1); \
    *((volatile uint64_t*) GEMMINI_RS2_ADDR) = (uint64_t) (rs2); \
    /* *((volatile uint32_t*) GEMMINI_RS2_ADDR) = (uint32_t) ((uint64_t) (rs2) & 0xFFFFFFFFULL); */ \
    /* *((volatile uint32_t*) (GEMMINI_RS2_ADDR + 4)) = (uint32_t) ((uint64_t) (rs2) >> 32); */ \
    pfence(); \
    /* gemmini_fence(); */ \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = instruction; \
    /* sprintf((char *) PRINT_BUF, "%llx %llx %d\n", rs1, rs2, funct); */ \
}

static void sp_tiled_matmul_full_spad_ws(const uint32_t A_sp_addr_start, const uint32_t B_sp_addr_start,
        const uint32_t D_sp_addr_start, const uint32_t C_dst_sp_addr_start,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act) {

  // const uint32_t A_sp_addr_start = 0;
  // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  // const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  // const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
  //   (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = 1; //full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  // const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  for (size_t k = 0; k < K; k++) {
    for (size_t j = 0; j < J; j++) {
      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;
          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = (k == 0); // no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }
          const size_t A_cols = DIM; // - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM; // - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM; // - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM; // - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM; // - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM; // - (i == I - 1 ? pad_I : 0);
          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, DIM, DIM);
          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          }
        }
        if (k == K - 1) {
          // Move-out C (if not normalizing)
          // if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
            const size_t rounded_j = (j / C_blocks) * C_blocks;
            const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;

            uint32_t C_dst_sp_addr = ((uint32_t) C_dst_sp_addr_start) + (i * J + rounded_j) * DIM; // * DIM * sizeof_C;

            const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
            const size_t cols = DIM; // blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM; // DIM - (i == I - 1 ? pad_I : 0);

            gemmini_extended_mvout_spad(C_dst_sp_addr, 1, rounded_C_sp_addr, cols, rows);
          // }
        }
      }
    }
  }
  pfence();
}


#endif
