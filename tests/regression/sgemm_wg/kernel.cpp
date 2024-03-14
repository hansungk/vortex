#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

#define BM 8
#define BN BM
#define BK 2
// #define TM (BM/BK)
// #define TN (BN/BK)
#define TM 2
#define TN 2

#define DEV_BARRIER_MMIO_BASE_ADDR 0xff003f00UL
#define CORES_PER_CLUSTER 2
#define BARRIER_STRIDE 4

void threadblock_barrier(unsigned int tid_in_threadblock, unsigned int barrier_id, unsigned int count) {
    vx_barrier(barrier_id, count);
    vx_fence();

    // vx_printf("========== barrier! barrier_id=%u, count=%u\n", barrier_id, count);

#if CORES_PER_CLUSTER != 0
    // this code doesn't work without the memory-mapped register implemented in
    // hardware, hence the #ifdef.

    if (tid_in_threadblock == 0) {
      volatile uint32_t *mmio = (volatile uint32_t *)(DEV_BARRIER_MMIO_BASE_ADDR);
      int core_id = vx_core_id();
      // FIXME: hardcoded
      const uint32_t barrier_stride = BARRIER_STRIDE;
      const uint32_t barrier_offset = barrier_stride * barrier_id;

      // wait for the barrier to be initialized
      while (mmio[barrier_offset + 1 + core_id] != 0);

      // signal internal-core synchronization done
      mmio[barrier_offset + 1 + core_id] = 1;

      // wait for other cores in the cluster to finish by waiting on the
      // all-synced read-only mmio reg
      while (mmio[barrier_offset] == 0);

      // need to signal that this core passed the barrier; otherwise, if we
      // reset this to 0 right away, the other core still waiting for the
      // barrier might never see the all-sync mmio reg as 1.
      mmio[barrier_offset + 1 + core_id] = 2;

      // // if this core is the last one passing the barrier, reset all per-core
      // // flags to 0 to get ready for the next barrier
      // bool all_passed = true;
      // for (int i = 0; i < CORES_PER_CLUSTER; i++) {
      //   // if (i == core_id) continue;
      //   // NOTE: this requires coherent access of store-to-load to the same
      //   // address
      //   if (mmio[barrier_offset + 1 + i] != 2) {
      //     all_passed = false;
      //     break;
      //   }
      // }
      // if (all_passed) {
      //   for (int i = 0; i < CORES_PER_CLUSTER; i++) {
      //     mmio[barrier_offset + 1 + i] = 0;
      //   }
      // }
    }

    vx_barrier(barrier_id, count);
#endif
}

void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threadblock_dim_x,
                              const uint32_t threadblock_dim_y,
                              const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,
                              const uint32_t threadblock_id_in_core,
                              float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

  // assumes NT == NW == matrix_dim
  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  // FIXME: Output block size is assumed to be square, i.e. BM == BN
  // const uint32_t BM = threadblock_dim_y;
  // const uint32_t BN = threadblock_dim_y;
  // const uint32_t BK = threadblock_dim_x;
  // constexpr uint32_t BM = 8;
  // constexpr uint32_t BN = 8;
  // constexpr uint32_t BK = 2;

  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;
  const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
  const uint32_t global_b_col = BN * threadblock_id_x + local_b_col;

  const uint32_t local_c_row = tid_in_threadblock / (BN / TN);
  const uint32_t local_c_col = tid_in_threadblock % (BN / TN);

  // each thread generates TM output element
  float reg_c[TM * TN] = { 0.0f };
  float reg_a[TM] = { 0.0f };
  float reg_b[TN] = { 0.0f };

  volatile float *local_a = sharedmem_per_threadblock;
  // const size_t local_a_elems = threadblock_dim_x * threadblock_dim_y;
  const size_t local_a_elems = (BM * BK);
  volatile float *local_b = sharedmem_per_threadblock + local_a_elems;

  constexpr uint32_t stride_a = (BM * BN) / BK / (TM * TN);
  constexpr uint32_t stride_b = (BM * BN) / BN / (TM * TN);

  for (uint32_t k = 0; k < dim_k; k += BK) {
    for (uint32_t load_offset = 0; load_offset < BM; load_offset += stride_a) {
      const uint32_t global_a_offset =
          dim_k * (global_a_row + load_offset) + (k + local_a_col);
      local_a[BK * (local_a_row + load_offset) + local_a_col] =
          A[global_a_offset];
    }
    for (uint32_t load_offset = 0; load_offset < BK; load_offset += stride_b) {
      const uint32_t global_b_offset =
          dim_n * (k + local_b_row + load_offset) + global_b_col;
      local_b[BN * (local_b_row + load_offset) + local_b_col] =
          B[global_b_offset];
    }

    threadblock_barrier(tid_in_threadblock, threadblock_id_in_core,
                        threadblock_dim_y);

    for (uint32_t local_k = 0; local_k < BK; local_k++) {
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

      // Compute multiple result elements (TM) per thread
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

    threadblock_barrier(tid_in_threadblock, threadblock_id_in_core,
                        threadblock_dim_y);
  }

#pragma GCC unroll TM
  for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
#pragma GCC unroll TN
    for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
      // NOTE use of local_b_row and global_b_col here
      C[dim_n * (BM * threadblock_id_y + TM * local_c_row + res_idx_m) +
        (BN * threadblock_id_x + TN * local_c_col + res_idx_n)] =
          reg_c[TN * res_idx_m + res_idx_n];
    }
  }
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

  const uint32_t threads_per_threadblock = (BM * BN) / (TM * TN);
  const uint32_t threadblocks_per_core =
      vx_num_threads() * vx_num_warps() / threads_per_threadblock;
  const uint32_t threadblock_dim_x = vx_num_threads();
  const uint32_t threadblock_dim_y = vx_num_warps() / threadblocks_per_core;
  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_core = threadblock_id % threadblocks_per_core;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;

  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR +
      (2 * BM * BK) * threadblock_id_in_core;
  thread_block_gemm(arg, tid_in_threadblock,
                    threadblock_dim_x, threadblock_dim_y, threadblock_id_x,
                    threadblock_id_y, threadblock_id_in_core,
                    sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  const uint32_t grid_size = arg->dim_m * arg->dim_n / (TM * TN);
  vx_spawn_tasks(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
