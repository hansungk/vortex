#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

inline void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock_x,
                              const uint32_t tid_in_threadblock_y,
                              const uint32_t threadblock_dim_x,
                              const uint32_t threadblock_dim_y,
                              const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y) {
  const float *global_a = (const float *)arg->addr_a;
  const float *global_b = (const float *)arg->addr_b;
  float *global_c = (float *)arg->addr_c;

  // assumes NT == NW == matrix_dim
  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  // FIXME: assumes local block size is square shape
  const uint32_t local_row = tid_in_threadblock_y;
  const uint32_t local_col = tid_in_threadblock_x;
  const uint32_t global_row = threadblock_id_y * threadblock_dim_y + local_row;
  const uint32_t global_col = threadblock_id_x * threadblock_dim_x + local_col;

  // each thread generates one output element
  float reg_c = 0.0f;

  for (uint32_t k = 0; k < dim_k; k += threadblock_dim_x) {
    float *local_a = (float *)DEV_SMEM_START_ADDR;
    size_t local_a_elems = threadblock_dim_x * threadblock_dim_y;
    float *local_b = (float *)DEV_SMEM_START_ADDR + local_a_elems;

    uint32_t offset_global_a = dim_k * global_row + (k + local_col);
    uint32_t offset_global_b = dim_n * (local_row + k) + global_col;
    // local_a: threadblock_dim_y rows, threadblock_dim_x cols
    // local_b: threadblock_dim_x rows, threadblock_dim_y cols
    // threadblock_dim_x == block_k, threadblock_dim_y == block_m == block_n
    local_a[threadblock_dim_x * local_row + local_col] = global_a[offset_global_a];
    local_b[threadblock_dim_y * local_col + local_row] = global_b[offset_global_b];

    vx_barrier(0, threadblock_dim_y);
    vx_fence();

    for (uint32_t local_k = 0; local_k < threadblock_dim_x; local_k++) {
      reg_c += local_a[threadblock_dim_x * local_row + local_k] *
               local_b[threadblock_dim_y * local_col + local_k];
    }

    vx_barrier(0, threadblock_dim_y);
    vx_fence();
  }

  global_c[dim_n * global_row + global_col] = reg_c;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t dim_n = arg->dim_n;
  int tid_x = task_id % dim_n;
  int tid_y = task_id / dim_n;

  const uint32_t threadblock_dim_x = vx_num_threads();
  const uint32_t threadblock_dim_y = vx_num_warps();
  const uint32_t threads_per_threadblock = threadblock_dim_x * threadblock_dim_y;
  const int threadblock_id = task_id / threads_per_threadblock;

  const uint32_t dim_n_in_blocks = dim_n / threadblock_dim_x;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;

  const int tid_in_threadblock_x = vx_thread_id();
  const int tid_in_threadblock_y = vx_warp_id() % threadblock_dim_y;
  thread_block_gemm(arg, tid_in_threadblock_x, tid_in_threadblock_y, threadblock_dim_x,
                    threadblock_dim_y, threadblock_id_x, threadblock_id_y);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  const uint32_t grid_size = arg->dim_m * arg->dim_n;
  vx_spawn_tasks(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
