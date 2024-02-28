#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

#define MAX_TM 4

void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t tid_in_threadblock_x,
                              const uint32_t tid_in_threadblock_y,
                              const uint32_t threadblock_dim_x,
                              const uint32_t threadblock_dim_y,
                              const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,
                              const uint32_t threadblock_id_in_core,
                              float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  // FIXME: Output block size is assumed to be square, i.e. BM == BN
  // const uint32_t BM = threadblock_dim_y;
  // const uint32_t BN = threadblock_dim_y;
  // const uint32_t BK = threadblock_dim_x;
  constexpr uint32_t BM = 8;
  constexpr uint32_t BN = 8;
  constexpr uint32_t BK = 4;
  constexpr uint32_t TM = 2;

  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;
  const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
  const uint32_t global_b_col = BN * threadblock_id_x + local_b_col;

  A += dim_k * BM * threadblock_id_y;
  B += BN * threadblock_id_x;
  C += dim_n * BM * threadblock_id_y + BN * threadblock_id_x;

  // each thread generates one output element
  float reg_c[MAX_TM] = { 0.0f };

  for (uint32_t k = 0; k < dim_k; k += BK) {
    float *local_a = sharedmem_per_threadblock;
    size_t local_a_elems = threadblock_dim_x * threadblock_dim_y;
    float *local_b = sharedmem_per_threadblock + local_a_elems;

    // NOTE: local_b is transposed to column-major to facilitate better memory
    // access.
    local_a[BK * local_a_row + local_a_col] = A[dim_k * local_a_row + local_a_col];
    local_b[BN * local_b_row + local_b_col] = B[dim_n * local_b_row + local_b_col];

    // Advance A and B block
    A += BK;
    B += dim_n * BK;

    vx_barrier(threadblock_id_in_core, threadblock_dim_y);
    vx_fence();

    for (uint32_t local_k = 0; local_k < BK; local_k++) {
      // Compute multiple result elements (TM) per thread
      const float local_b_tmp = local_b[BN * local_k + local_b_col];
#pragma GCC unroll 1
      for (uint32_t result_idx = 0; result_idx < TM; result_idx++) {
        reg_c[result_idx] +=
            local_a[BK * (TM * local_b_row + result_idx) + local_k] *
            local_b_tmp;
      }
    }

    vx_barrier(threadblock_id_in_core, threadblock_dim_y);
    vx_fence();
  }

#pragma GCC unroll 1
  for (uint32_t result_idx = 0; result_idx < TM; result_idx++) {
    C[dim_n * (TM * local_b_row + result_idx) + local_b_col] = reg_c[result_idx];
  }
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

  const uint32_t threadblocks_per_core = 1;
  const uint32_t threadblock_dim_x = vx_num_threads();
  const uint32_t threadblock_dim_y = vx_num_warps() / threadblocks_per_core;
  const uint32_t threads_per_threadblock = threadblock_dim_x * threadblock_dim_y;
  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_core = threadblock_id % threadblocks_per_core;

  const int tid_in_threadblock = task_id % threads_per_threadblock;
  const int tid_in_threadblock_x = vx_thread_id();
  const int tid_in_threadblock_y = vx_warp_id() % threadblock_dim_y;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t BN = 8;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;
  // const int threadblock_id_x = dim_n / threadblock_dim_x;
  // const int threadblock_id_y = dim_m / threadblock_dim_y / 1;

  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR +
      (2 * threads_per_threadblock) * threadblock_id_in_core;
  thread_block_gemm(arg, tid_in_threadblock, tid_in_threadblock_x, tid_in_threadblock_y,
                    threadblock_dim_x, threadblock_dim_y, threadblock_id_x,
                    threadblock_id_y, threadblock_id_in_core,
                    sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  const uint32_t grid_size = arg->dim_m * arg->dim_n / 2;
  vx_spawn_tasks(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
