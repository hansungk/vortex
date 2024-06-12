#ifndef _UTIL_H_
#define _UTIL_H_

#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define NUM_LANES 8

// Constraints on parameters:
// * Memory:
//   (BM + BN) * BK * sizeof(float) <= sharedmem size.
//   BM * BK == BN * BK >= threadblock size >= NT * CORES_PER_CLUSTER
//     When larger, the kernel runs a sequential loop to read into sharedmem;
//     but smaller case is not handled.
// * Compute:
//   ( M* N) / (TM*TN) == grid size >= NC*NW*NT
//   (BM*BN) / (TM*TN) == threadblock size < NT * NW * CORES_PER_CLUSTER
//   (BM*BN) / (TM*TN) == threadblock size >= NT * CORES_PER_CLUSTER
// * Combining BM * BK >= (BM*BN) / (TM*TN) == threadblock yields
//   BM <= BK*TM*TN
#define BM 32
#define BN 32
#define BK 32
#define WM 16
#define WN 8
#define TCM 8
#define TCN 8
#define TCK 8
#define WMITER (WM / TCM)
#define WNITER (WN / TCN)
#define ELEM_PER_THREAD (WMITER * WNITER * (TCM * TCN) / NUM_LANES)

// number of loop around the inner 0..TCK..BK loop to simulate perfect-DRAM
// scenario
#define BK_LOOP 1
#define TRANSPOSE_AS 1
// GMEM_COALESCED sets bank conflict-free accesses for
// 1: GMEM loads of A matrix
// 0: SMEM stores of A matrix
#define GMEM_COALESCED_A 1
#define GEMMINI_DMA 0
#if SMEM_SIZE != 0x4000
#error Currently only supports 16K spad
#endif
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff001000)
#define SMEM_ADDR_Q2 ((float * const) 0xff002000)
#define SMEM_ADDR_Q3 ((float * const) 0xff003000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x80
#define SPAD_ADDR_Q2 0x100
#define SPAD_ADDR_Q3 0x180

// FIXME: NUM_THREADS and NUM_WARPS hardcoded
#if ((BM * BN / ELEM_PER_THREAD) > (CORES_PER_CLUSTER * 8 * 8))
#error "threadblock size too big for cluster"
#endif

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
  if constexpr (NUM_LANES == 32) {
    map_operand_32lanes(tid, row, col);
  } else if constexpr (NUM_LANES == 8) {
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
  if constexpr (NUM_LANES == 32) {
    map_c_32lanes(tid, row, col);
  } else if constexpr (NUM_LANES == 8) {
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
inline void vx_wmma_load_a(volatile float *smem_A, const int local_k,
                  const int warp_row, const int wm_iter, const int thread_in_warp) {
  const int tid = thread_in_warp;
  const int tg = tid / 4;

  // TODO: this is duplicately computed between vx_wmma_load_a and vx_wmma_load_b
  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  constexpr int smem_A_rows = BM;
  constexpr int smem_A_cols = BK;
  constexpr int smem_AS_rows = BK;
  constexpr int smem_AS_cols = BM;

  if constexpr (!TRANSPOSE_AS) {
    // int A_offset = (WM * warp_row + TCM * wm_iter + row) * smem_A_cols;

    // @perf: bank conflicts
    // f8-f15 stores a single row of A
    volatile float *smem_addr;
    smem_addr = &smem_A[(WM * warp_row + TCM * wm_iter + row) * smem_A_cols + local_k];
    asm volatile("flw  f0, %0(%1)" ::"i"(0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" ::"i"(1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" ::"i"(2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" ::"i"(3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" ::"i"(4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" ::"i"(5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" ::"i"(6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" ::"i"(7 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f0, %0" ::"m"(smem_A[A_offset + (local_k + 0)]));
    // asm volatile("flw  f1, %0" ::"m"(smem_A[A_offset + (local_k + 1)]));
    // asm volatile("flw  f2, %0" ::"m"(smem_A[A_offset + (local_k + 2)]));
    // asm volatile("flw  f3, %0" ::"m"(smem_A[A_offset + (local_k + 3)]));
    // asm volatile("flw  f4, %0" ::"m"(smem_A[A_offset + (local_k + 4)]));
    // asm volatile("flw  f5, %0" ::"m"(smem_A[A_offset + (local_k + 5)]));
    // asm volatile("flw  f6, %0" ::"m"(smem_A[A_offset + (local_k + 6)]));
    // asm volatile("flw  f7, %0" ::"m"(smem_A[A_offset + (local_k + 7)]));
  } else {
    // transposed A
    // f8-f15 stores a single row of A
    volatile float *smem_addr;
    smem_addr = &smem_A[((local_k + 0) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row];
    asm volatile("flw  f0, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" :: "i"(smem_AS_cols * 4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" :: "i"(smem_AS_cols * 5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" :: "i"(smem_AS_cols * 6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" :: "i"(smem_AS_cols * 7 * sizeof(float)), "r"(smem_addr));

    // asm volatile("flw  f0, %0" ::"m"(smem_A[((local_k + 0) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f1, %0" ::"m"(smem_A[((local_k + 1) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f2, %0" ::"m"(smem_A[((local_k + 2) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f3, %0" ::"m"(smem_A[((local_k + 3) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f4, %0" ::"m"(smem_A[((local_k + 4) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f5, %0" ::"m"(smem_A[((local_k + 5) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f6, %0" ::"m"(smem_A[((local_k + 6) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f7, %0" ::"m"(smem_A[((local_k + 7) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
  }
}

// `local_k` is assumed to be multiple of TCK
inline void vx_wmma_load_b(volatile float *smem_B, const int local_k,
                           const int warp_col, const int wn_iter,
                           const int thread_in_warp) {
  const int tid = thread_in_warp;
  const int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  constexpr int smem_B_rows = BK;
  constexpr int smem_B_cols = BN;

  // f8-f15 stores a single column of B
  volatile float *smem_addr;
  smem_addr = &smem_B[((local_k + 0) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col];
  asm volatile("flw  f8, %0(%1)" :: "i"(smem_B_cols * 0 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw  f9, %0(%1)" :: "i"(smem_B_cols * 1 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f10, %0(%1)" :: "i"(smem_B_cols * 2 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f11, %0(%1)" :: "i"(smem_B_cols * 3 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f12, %0(%1)" :: "i"(smem_B_cols * 4 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f13, %0(%1)" :: "i"(smem_B_cols * 5 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f14, %0(%1)" :: "i"(smem_B_cols * 6 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f15, %0(%1)" :: "i"(smem_B_cols * 7 * sizeof(float)), "r"(smem_addr));

  // asm volatile("flw  f8, %0" ::"m"(smem_B[((local_k + 0) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw  f9, %0" ::"m"(smem_B[((local_k + 1) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f10, %0" ::"m"(smem_B[((local_k + 2) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f11, %0" ::"m"(smem_B[((local_k + 3) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f12, %0" ::"m"(smem_B[((local_k + 4) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f13, %0" ::"m"(smem_B[((local_k + 5) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f14, %0" ::"m"(smem_B[((local_k + 6) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f15, %0" ::"m"(smem_B[((local_k + 7) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
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

  float *global_offset_C = C +
                           (BM * threadblock_id_y) * dim_n +
                           BN * threadblock_id_x;

  // @perf: this likely causes a lot of gmem bank conflicts
  if (wm_iter == 0) {
    volatile float *gmem_addr = &global_offset_C[dim_n * (local_row + 0) + (local_col + 0)];
    volatile float *gmem_addr_tmp = gmem_addr + (2 * dim_n);
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
    volatile float *gmem_addr = &global_offset_C[dim_n * (local_row + 0) + (local_col + 0)];
    volatile float *gmem_addr_tmp = gmem_addr + (2 * dim_n);
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
