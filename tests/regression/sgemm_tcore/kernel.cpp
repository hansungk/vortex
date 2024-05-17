#define RISCV_CUSTOM3   0x7B

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

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
#define BM 16
#define BN 16
#define BK 8
#define TCM 8
#define TCN 8
#define TCK 8
#define WM 8
#define WN 8
#define WMITER (WM / TCM)
#define WNITER (WN / TCN)
#define TM 1
#define TN ((TCM * TCN) / NUM_LANES / TM)
// #define TN 1
#define ELEM_PER_THREAD (WMITER * WNITER * TM * TN)

#define USE_TENSOR_CORE 1
#define TC_SINGLE_WARP 0
// number of loop around the inner 0..TCK..BK loop to simulate perfect-DRAM
// scenario
#define BK_LOOP 16
#define TRANSPOSE_AS 1

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

inline void vx_wmma() {
  asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
}

// `local_k` is assumed to be multiple of TCK
inline void vx_wmma_load(volatile float *smem_A, volatile float *smem_B, const int local_k,
                  const int warp_col, const int warp_row, const int wn_iter,
                  const int wm_iter, const int thread_in_warp) {
  int tid = thread_in_warp;
  int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  int smem_A_rows = BM;
  int smem_A_cols = BK;
  int smem_B_rows = BK;
  int smem_B_cols = BN;

  if constexpr (!TRANSPOSE_AS) {
    int A_offset = (row + WM * warp_row + TCM * wm_iter) * smem_A_cols;

    // @perf: bank conflicts
    asm volatile("flw f0, %0" ::"m"(smem_A[A_offset + (local_k + 0)]));
    asm volatile("flw f1, %0" ::"m"(smem_A[A_offset + (local_k + 1)]));
    asm volatile("flw f2, %0" ::"m"(smem_A[A_offset + (local_k + 2)]));
    asm volatile("flw f3, %0" ::"m"(smem_A[A_offset + (local_k + 3)]));
    asm volatile("flw f4, %0" ::"m"(smem_A[A_offset + (local_k + 4)]));
    asm volatile("flw f5, %0" ::"m"(smem_A[A_offset + (local_k + 5)]));
    asm volatile("flw f6, %0" ::"m"(smem_A[A_offset + (local_k + 6)]));
    asm volatile("flw f7, %0" ::"m"(smem_A[A_offset + (local_k + 7)]));
  } else {
    // transposed A
    asm volatile("flw  f0, %0" ::"m"(smem_A[((local_k + 0) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f1, %0" ::"m"(smem_A[((local_k + 1) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f2, %0" ::"m"(smem_A[((local_k + 2) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f3, %0" ::"m"(smem_A[((local_k + 3) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f4, %0" ::"m"(smem_A[((local_k + 4) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f5, %0" ::"m"(smem_A[((local_k + 5) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f6, %0" ::"m"(smem_A[((local_k + 6) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
    asm volatile("flw  f7, %0" ::"m"(smem_A[((local_k + 7) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
// #pragma GCC unroll 8
//     for (int i = 0; i < 8; i++) {
//       asm volatile("flw  f0, %0" ::"m"(smem_A[((local_k + i) * smem_A_rows) + (WM * warp_row + TCM * wm_iter) + row]));
//     }
  }

  asm volatile("flw  f8, %0" ::"m"(smem_B[((local_k + 0) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw  f9, %0" ::"m"(smem_B[((local_k + 1) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw f10, %0" ::"m"(smem_B[((local_k + 2) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw f11, %0" ::"m"(smem_B[((local_k + 3) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw f12, %0" ::"m"(smem_B[((local_k + 4) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw f13, %0" ::"m"(smem_B[((local_k + 5) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw f14, %0" ::"m"(smem_B[((local_k + 6) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  asm volatile("flw f15, %0" ::"m"(smem_B[((local_k + 7) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
}

inline void initialize_C() {
  // initialize C to zeros
  asm volatile("fmv.w.x f16, x0");
  asm volatile("fmv.w.x f17, x0");
  asm volatile("fmv.w.x f18, x0");
  asm volatile("fmv.w.x f19, x0");
  asm volatile("fmv.w.x f20, x0");
  asm volatile("fmv.w.x f21, x0");
  asm volatile("fmv.w.x f22, x0");
  asm volatile("fmv.w.x f23, x0");
}

inline void write_results(volatile float *local_warp_results,
                          int thread_in_warp, int warp_col, int warp_row,
                          int wn_iter, int wm_iter, int dim_m, int dim_n,
                          float *C, int threadblock_id_x,
                          int threadblock_id_y) {
  int tid = thread_in_warp;
  int tg = tid / 4;

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
  asm volatile ("fsw f16, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 0)]));
  asm volatile ("fsw f17, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 1)]));
  asm volatile ("fsw f18, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 0)]));
  asm volatile ("fsw f19, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 1)]));
  asm volatile ("fsw f20, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 4)]));
  asm volatile ("fsw f21, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 5)]));
  asm volatile ("fsw f22, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 4)]));
  asm volatile ("fsw f23, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 5)]));
}

void threadblock_barrier(unsigned int tid_in_threadblock, unsigned int barrier_id, unsigned int count) {
    vx_fence();
    vx_barrier(barrier_id, count);
}

void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threadblock_dim_x,
                              const uint32_t threadblock_dim_y,
                              const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,
                              const uint32_t threadblock_id_in_cluster,
                              float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_as_row = tid_in_threadblock / BM;
  const uint32_t local_as_col = tid_in_threadblock % BM;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;

  const uint32_t local_c_row = tid_in_threadblock / (BN / TN);
  const uint32_t local_c_col = tid_in_threadblock % (BN / TN);

  // each thread generates TM output element
  float reg_c[TM * TN] = { 0.0f };
  float reg_a[TM] = { 0.0f };
  float reg_b[TN] = { 0.0f };

  const uint32_t warp_in_threadblock = tid_in_threadblock / NUM_LANES;
  const uint32_t warp_row = warp_in_threadblock / (BN / WN);
  const uint32_t warp_col = warp_in_threadblock % (BN / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_LANES;

  volatile float *local_a = sharedmem_per_threadblock;
  // const size_t local_a_elems = threadblock_dim_x * threadblock_dim_y;
  const size_t local_a_elems = (BM * BK);
  volatile float *local_b = sharedmem_per_threadblock + local_a_elems;
  const size_t local_b_elems = (BK * BN);
  volatile float *local_warp_results =
      local_b + local_b_elems + (warp_in_threadblock * TCM * TCN);

  // clear out C
  initialize_C();

#pragma GCC unroll 1
  for (uint32_t k = 0; k < dim_k; k += BK) {
    // Data move from GMEM to SMEM
    //
    // Make sure global offset values for A and B are contiguous between
    // neighboring threads to ensure GMEM coalescing.
    //
    // TODO: Sharedmem swizzling is important here
    if constexpr (!TRANSPOSE_AS) {
      const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
      // number of rows a full TB can read at a time
      constexpr uint32_t row_stride_a = (BM * BN) / ELEM_PER_THREAD / BK;
#pragma GCC unroll 1
      for (uint32_t load_offset = 0; load_offset < BM; load_offset += row_stride_a) {
        const uint32_t global_a_offset =
            dim_k * (global_a_row + load_offset) + (k + local_a_col);
        // NOTE: all threads in TB  will do this load; make sure this is not
        // out-of-bounds of BM*BK
        local_a[BK * (local_a_row + load_offset) + local_a_col] =
            A[global_a_offset];
      }
    } else {
      const uint32_t global_a_row = BM * threadblock_id_y + local_as_col;
      constexpr uint32_t row_stride_a = (BM * BN) / ELEM_PER_THREAD / BM;
#pragma GCC unroll 1
      for (uint32_t load_offset = 0; load_offset < BK; load_offset += row_stride_a) {
        const uint32_t global_a_offset =
            dim_k * (global_a_row + load_offset) + (k + local_as_row);
        local_a[BM * (local_as_row + load_offset) + local_as_col] =
            A[global_a_offset];
      }
    }

  constexpr uint32_t row_stride_b = (BM * BN) / ELEM_PER_THREAD / BN;
  const uint32_t global_b_col = BN * threadblock_id_x + local_b_col;
#pragma GCC unroll 1
    for (uint32_t load_offset = 0; load_offset < BK; load_offset += row_stride_b) {
      const uint32_t global_b_offset =
          dim_n * (k + local_b_row + load_offset) + global_b_col;
      local_b[BN * (local_b_row + load_offset) + local_b_col] =
          B[global_b_offset];
    }

    threadblock_barrier(tid_in_threadblock, threadblock_id_in_cluster,
                        threadblock_dim_y);

#if USE_TENSOR_CORE
// #pragma GCC unroll 1
    for (int i = 0; i < BK_LOOP; i++) {
      // @perf: this loop spills to stack a lot because of all the flws in vx_wmma_load
#pragma GCC unroll 1
      for (uint32_t local_k = 0; local_k < BK; local_k += TCK) {
        // perform wmma
        // vx_wmma_load(local_a, local_b, warp_x, warp_y, tid_in_warp);
        // FIXME: If multiple warps try to issue to Tensor Core at the same time,
        // does one stall the other?
        // FIXME: this is wrong!! need separate accumulation register for
        // WM/WN_ITERS
#pragma GCC unroll 1
        for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll 1
          for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
#if TC_SINGLE_WARP
            if (warp_in_threadblock == 0) {
#endif
              // if ((threadblock_id_in_cluster % 2) == 0) {
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              //     asm volatile("addi a0, a0, 0");
              // }
              // SMEM -> RF
              vx_wmma_load(local_a, local_b, local_k, warp_col, warp_row, wn_iter,
                  wm_iter, tid_in_warp);
              // compute
              vx_wmma();
#if TC_SINGLE_WARP
            }
#endif
          }
        }
      }
    }

#else

    // Compute single tile*tile matmul
#pragma GCC unroll 4
    for (uint32_t local_k = 0; local_k < BK; local_k++) {
      // First, pump data from SMEM->RF
#pragma GCC unroll TM
      for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
        reg_a[res_idx_m] =
            local_a[BK * (TM * local_c_row + res_idx_m) + local_k];
      }
#pragma GCC unroll TN
      for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
        reg_b[res_idx_n] =
            local_b[BN * local_k + (TN * local_c_col + res_idx_n)];
      }

      // Next, compute multiple result elements (TM*TN) by reusing data in RF
#pragma GCC unroll TM
      for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
#pragma GCC unroll TN
        for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
          // NOTE use of local_b_row
          reg_c[TN * res_idx_m + res_idx_n] +=
              reg_a[res_idx_m] * reg_b[res_idx_n];
          // reg_c[TN * res_idx_m + res_idx_n] +=
          //     local_a[BK * (TM * local_c_row + res_idx_m) + local_k] *
          //     local_b[BN * local_k + (TN * local_c_col + res_idx_n)];
        }
      }
    }
#endif

    threadblock_barrier(tid_in_threadblock, threadblock_id_in_cluster,
                        threadblock_dim_y);
  }

#if USE_TENSOR_CORE
#pragma GCC unroll 1
  for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll 1
    for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
#if TC_SINGLE_WARP
      if (warp_in_threadblock == 0) {
#endif
        write_results(local_warp_results, tid_in_warp, warp_col, warp_row,
                      wn_iter, wm_iter, dim_m, dim_n, C, threadblock_id_x,
                      threadblock_id_y);
#if TC_SINGLE_WARP
      }
#endif
    }
  }

#else

  // Store result data from RF to GMEM
#pragma GCC unroll TM
  for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
#pragma GCC unroll TN
    for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
      C[dim_n * (BM * threadblock_id_y + TM * local_c_row + res_idx_m) +
        (BN * threadblock_id_x + TN * local_c_col + res_idx_n)] =
          reg_c[TN * res_idx_m + res_idx_n];
    }
  }
#endif

}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

  const uint32_t threads_per_threadblock = (BM * BN) / (ELEM_PER_THREAD);
#ifdef RADIANCE
  const uint32_t threadblocks_per_core = CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps() /
                                         threads_per_threadblock;
#else
  const uint32_t threadblocks_per_core =
      vx_num_threads() * vx_num_warps() / threads_per_threadblock;
#endif
  const uint32_t threadblock_dim_x = vx_num_threads();
  const uint32_t threadblock_dim_y = vx_num_warps() / threadblocks_per_core;
  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_cluster = threadblock_id % threadblocks_per_core;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  // FIXME: 4* is unnecessary; being safe for overlaps
  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR + (4 * BM * BK) * threadblock_id_in_cluster;
  thread_block_gemm(arg, tid_in_threadblock, threadblock_dim_x,
                    threadblock_dim_y, threadblock_id_x, threadblock_id_y,
                    threadblock_id_in_cluster, sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  const uint32_t grid_size = arg->dim_m * arg->dim_n / ELEM_PER_THREAD;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
