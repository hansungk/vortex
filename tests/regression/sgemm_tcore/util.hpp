#ifndef _UTIL_H_
#define _UTIL_H_

#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "include/gemmini.h"
#include "gemmini_mmio.h"

// Constraints on parameters:
// * Memory:
//   (BM + BN) * BK * sizeof(T) <= sharedmem size.
//   BM * BK == BN * BK >= threadblock size >= NT * CORES_PER_CLUSTER
//     When larger, the kernel runs a sequential loop to read into sharedmem;
//     but smaller case is not handled.
// * Compute:
//   ( M* N) / (TM*TN) == grid size >= NC*NW*NT
//   (BM*BN) / (TM*TN) == threadblock size < NT * NW * CORES_PER_CLUSTER
//   (BM*BN) / (TM*TN) == threadblock size >= NT * CORES_PER_CLUSTER
// * Combining BM * BK >= (BM*BN) / (TM*TN) == threadblock yields
//   BM <= BK*TM*TN
#define BM 64
#define BN 64
#define BK 128
#define WM 16
#define WN 8
#define TCM 8
#define TCN 8
#define TCK 16
#define WMITER (WM / TCM)
#define WNITER (WN / TCN)
#define ELEM_PER_THREAD (WMITER * WNITER * (TCM * TCN) / NUM_THREADS)

// number of loop around the inner 0..TCK..BK loop to simulate perfect-DRAM
// scenario
#define BK_LOOP 1
// Whether to transpose smem A tile at GMEM->SMEM (produce), or SMEM->RF
// (consume).  This is because the tensor core expects the A tile to be stored
// in column-major order in SMEM, whereas it will be ultimately stored in
// row-major in the RF.
//
// For correctness, only one of either should be 1.  E.g., PRODUCE 1 CONSUME 0
// generates the NN kernel where both A and B are stored row-major in GMEM.
// To model the case where the A matrix is already stored column-major in GMEM,
// set both to 0.
#define TRANSPOSE_AT_PRODUCE 0
#define TRANSPOSE_AT_CONSUME 0
// GMEM_COALESCED: When TRANSPOSE_AT_PRODUCE == 1 (i.e. transpose at
// GMEM->SMEM), determines whether we do bank-conflict-free accesses for
//   1: GMEM loads of A matrix, or
//   0: SMEM stores of A matrix.
//
// Usually, GMEM_COALESCED==1 yields better performance since the memory
// behavior of GMEM is more sensitive to bank conflicts.
#define GMEM_COALESCED_A 1

// "fake" fp16 type that only has the correct data width.
using float16_t = uint16_t;

inline constexpr void map_operand_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // Figure 7(a) in paper
  // row  0~ 3: threadgroups 0 and 2
  // row  4~ 7: threadgroups 4 and 6
  // row  8~11: threadgroups 1 and 3
  // row 12~15: threadgroups 5 and 7
  row = tid % 4;
  row += (tg * 8) % 16;
  row += (tg / 4) * 4;

  // B (column major)
  // NOTE: Matrix B mapping in Figure 7(a) is incorrect; below is the
  // corrected mapping:
  // col  0~ 3: threadgroups 0 and 1
  // col  4~ 7: threadgroups 4 and 5
  // col  8~11: threadgroups 2 and 3
  // col 12~15: threadgroups 6 and 7
  col = tid % 4;
  col += ((tg % 4) / 2) * 8;
  col += (tg / 4) * 4;
}

inline constexpr void map_operand_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // row  0~ 3: threadgroup 0
  // row  4~ 7: threadgroup 1
  row = tid % 4;
  row += tg * 4;

  // B (column major)
  // col  0~ 3: threadgroup 0
  // col  4~ 7: threadgroup 1
  col = tid % 4;
  col += tg * 4;
}

inline constexpr void map_operand(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_operand_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_operand_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

inline constexpr void map_c_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  // Figure 7(b), left
  col = ((tg % 4) / 2) * 8;
  row = (tg * 8) % 16;
  row += (tg / 4) * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  col = 0;
  row = tg * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_c_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_c_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

#define RISCV_CUSTOM3   0x7B

inline void vx_wmma(const int dest_reg) {
  if (dest_reg == 0) {
    asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
  } else {
    asm volatile (".insn r %0, 0, 0, x1, x0, x0" :: "i"(RISCV_CUSTOM3));
  }
}

// `local_k` is assumed to be multiple of TCK
template <typename T>
inline void vx_wmma_load_a(volatile const T *smem_A, const int local_k,
                           const int warp_row, const int wm_iter,
                           const int thread_in_warp) {
  const int tid = thread_in_warp;
  const int tg = tid / 4;

  // @perf: this is duplicately computed in vx_wmma_load_a and vx_wmma_load_b
  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  // In fp16 mode, bit-pack two fp16 elements into each fp32 element, and do
  // data movement at the fp32 granularity.  Assuming that the matrix is stored
  // row-major in GMEM, the packed fp16 pairs belong to the same row,
  // neighboring columns; therefore, it essentially becomes equivalent to
  // moving a fp32 matrix whose column dimensions (dim_k/BK/k) are compressed
  // by a factor of two.
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  constexpr int BK_adjusted = BK / packed_factor;
  constexpr int BM_adjusted = BM / packed_factor;
  const int local_k_adjusted = local_k / packed_factor;

  constexpr int smem_A_rows = BM;
  constexpr int smem_A_cols = BK_adjusted;
  constexpr int smem_AS_rows = BK_adjusted;
  constexpr int smem_AS_cols = BM;
  // constexpr int smem_AS_rows = BK;
  // constexpr int smem_AS_cols = BM_adjusted;

  if constexpr (TRANSPOSE_AT_CONSUME) {
    // int A_offset = (WM * warp_row + TCM * wm_iter + row) * smem_A_cols;

    // @perf: bank conflicts
    // f8-f15 stores a single row of A
    const volatile uint8_t *smem_addr;
    smem_addr = reinterpret_cast<const volatile uint8_t *>(
        &reinterpret_cast<const volatile float *>(
            smem_A)[(WM * warp_row + TCM * wm_iter + row) * smem_A_cols +
                    local_k /* FIXME: adjust for fp16? */]);
    // step to the next column
    // threads read from different rows; bank conflicts
    asm volatile("flw  f0, %0(%1)" ::"i"(0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" ::"i"(1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" ::"i"(2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" ::"i"(3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" ::"i"(4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" ::"i"(5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" ::"i"(6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" ::"i"(7 * sizeof(float)), "r"(smem_addr));
  } else {
    // read smem A tile as-is; bank-conflict-free AS load
    // smem A tile is stored column-major
    // f8-f15 stores a single row of A
    const volatile uint8_t *smem_addr;
    smem_addr = reinterpret_cast<const volatile uint8_t *>(
        &reinterpret_cast<const volatile float *>(
            smem_A)[((local_k_adjusted + 0) * smem_AS_cols) +
                    (WM * warp_row + TCM * wm_iter) + row]);
    // step to the next row
    // threads read from different columns; no bank conflicts
    asm volatile("flw  f0, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" :: "i"(smem_AS_cols * 4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" :: "i"(smem_AS_cols * 5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" :: "i"(smem_AS_cols * 6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" :: "i"(smem_AS_cols * 7 * sizeof(float)), "r"(smem_addr));
  }
}

// `local_k` is assumed to be multiple of TCK
template <typename T>
inline void vx_wmma_load_b(const volatile T *smem_B, const int local_k,
                           const int warp_col, const int wn_iter,
                           const int thread_in_warp) {
  const int tid = thread_in_warp;
  const int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  // see comment in vx_wmma_load_a
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  constexpr int BK_adjusted = BN / packed_factor;
  constexpr int BN_adjusted = BN / packed_factor;
  const int local_k_adjusted = local_k / packed_factor;

  // constexpr int smem_B_rows = BK;
  // constexpr int smem_B_cols = BN_adjusted;
  constexpr int smem_B_rows = BK_adjusted;
  constexpr int smem_B_cols = BN;

  // f8-f15 stores a single column of B
  const volatile uint8_t *smem_addr;
  smem_addr = reinterpret_cast<const volatile uint8_t *>(
      &reinterpret_cast<const volatile float *>(
          smem_B)[((local_k_adjusted + 0) * smem_B_cols) +
                  (WN * warp_col + TCN * wn_iter) + col]);
  // step to the next row
  // threads read from different columns; no bank conflicts
  asm volatile("flw  f8, %0(%1)" :: "i"(smem_B_cols * 0 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw  f9, %0(%1)" :: "i"(smem_B_cols * 1 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f10, %0(%1)" :: "i"(smem_B_cols * 2 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f11, %0(%1)" :: "i"(smem_B_cols * 3 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f12, %0(%1)" :: "i"(smem_B_cols * 4 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f13, %0(%1)" :: "i"(smem_B_cols * 5 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f14, %0(%1)" :: "i"(smem_B_cols * 6 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f15, %0(%1)" :: "i"(smem_B_cols * 7 * sizeof(float)), "r"(smem_addr));
}

inline void initialize_C(const int dest_reg) {
  // initialize C to zeros
  if (dest_reg == 0) {
    asm volatile("fmv.w.x f16, x0");
    asm volatile("fmv.w.x f17, x0");
    asm volatile("fmv.w.x f18, x0");
    asm volatile("fmv.w.x f19, x0");
    asm volatile("fmv.w.x f20, x0");
    asm volatile("fmv.w.x f21, x0");
    asm volatile("fmv.w.x f22, x0");
    asm volatile("fmv.w.x f23, x0");
  } else {
    asm volatile("fmv.w.x f24, x0");
    asm volatile("fmv.w.x f25, x0");
    asm volatile("fmv.w.x f26, x0");
    asm volatile("fmv.w.x f27, x0");
    asm volatile("fmv.w.x f28, x0");
    asm volatile("fmv.w.x f29, x0");
    asm volatile("fmv.w.x f30, x0");
    asm volatile("fmv.w.x f31, x0");
  }
}

inline void write_results(const int thread_in_warp, const int warp_col,
                          const int warp_row, const int wn_iter,
                          const int wm_iter, const int dim_n,
                          float *C, const int threadblock_id_x,
                          const int threadblock_id_y) {
  int tid = thread_in_warp;

  // these are [0, TCM/TCN)
  int tid_row = 0;
  int tid_col = 0;
  map_c(tid, tid_row, tid_col);

  int local_row = (WM * warp_row + TCM * wm_iter) + tid_row;
  int local_col = (WN * warp_col + TCN * wn_iter) + tid_col;

  float *global_offset_C =
      C + (BM * threadblock_id_y) * dim_n + BN * threadblock_id_x;

  // @perf: this likely causes a lot of gmem bank conflicts
  if (wm_iter == 0) {
    volatile uint8_t *gmem_addr = reinterpret_cast<volatile uint8_t *>(
        &global_offset_C[dim_n * (local_row + 0) + (local_col + 0)]);
    volatile uint8_t *gmem_addr_tmp = gmem_addr + (2 * dim_n) * sizeof(float);
    asm volatile ("fsw f16, %0(%1)" :: "i"(0 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f17, %0(%1)" :: "i"(1 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f18, %0(%1)" :: "i"(0 * sizeof(float)), "r"(gmem_addr_tmp));
    asm volatile ("fsw f19, %0(%1)" :: "i"(1 * sizeof(float)), "r"(gmem_addr_tmp));
    asm volatile ("fsw f20, %0(%1)" :: "i"(4 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f21, %0(%1)" :: "i"(5 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f22, %0(%1)" :: "i"(4 * sizeof(float)), "r"(gmem_addr_tmp));
    asm volatile ("fsw f23, %0(%1)" :: "i"(5 * sizeof(float)), "r"(gmem_addr_tmp));
    // asm volatile ("fsw f16, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 0)]));
    // asm volatile ("fsw f17, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 1)]));
    // asm volatile ("fsw f18, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 0)]));
    // asm volatile ("fsw f19, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 1)]));
    // asm volatile ("fsw f20, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 4)]));
    // asm volatile ("fsw f21, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 5)]));
    // asm volatile ("fsw f22, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 4)]));
    // asm volatile ("fsw f23, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 5)]));
  } else {
    volatile uint8_t *gmem_addr = reinterpret_cast<volatile uint8_t *>(
        &global_offset_C[dim_n * (local_row + 0) + (local_col + 0)]);
    volatile uint8_t *gmem_addr_tmp = gmem_addr + (2 * dim_n) * sizeof(float);
    asm volatile ("fsw f24, %0(%1)" :: "i"(0 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f25, %0(%1)" :: "i"(1 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f26, %0(%1)" :: "i"(0 * sizeof(float)), "r"(gmem_addr_tmp));
    asm volatile ("fsw f27, %0(%1)" :: "i"(1 * sizeof(float)), "r"(gmem_addr_tmp));
    asm volatile ("fsw f28, %0(%1)" :: "i"(4 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f29, %0(%1)" :: "i"(5 * sizeof(float)), "r"(gmem_addr));
    asm volatile ("fsw f30, %0(%1)" :: "i"(4 * sizeof(float)), "r"(gmem_addr_tmp));
    asm volatile ("fsw f31, %0(%1)" :: "i"(5 * sizeof(float)), "r"(gmem_addr_tmp));
  }
}

inline void threadblock_barrier(const uint32_t barrier_id, const uint32_t count) {
  vx_fence();
  vx_barrier(barrier_id, count);
}

#endif
