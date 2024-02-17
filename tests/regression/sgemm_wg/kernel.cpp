#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const float *global_a = (const float *)arg->addr_a;
  const float *global_b = (const float *)arg->addr_b;
  float *global_c = (float *)arg->addr_c;

  // assumes NT == NW == matrix_dim
  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;
  const uint32_t block_dim = vx_num_warps();
  const uint32_t local_row = vx_warp_id();
  const uint32_t local_col = vx_thread_id();

  // each thread generates one output element
  float reg_c = 0.0f;

  for (uint32_t k = 0; k < dim_k; k += block_dim) {
    float *local_a = (float *)DEV_SMEM_START_ADDR;
    float *local_b = (float *)DEV_SMEM_START_ADDR + (block_dim * block_dim);

    // FIXME: assumes local block size is square shape
    // TODO: "local_row" should be global_row
    uint32_t offset_global_a = dim_k * local_row + (k + local_col);
    uint32_t offset_global_b = dim_n * (local_row + k) + local_col;
    local_a[block_dim * local_row + local_col] = global_a[offset_global_a];
    local_b[block_dim * local_row + local_col] = global_b[offset_global_b];

    vx_barrier(0, vx_num_warps());
    vx_fence();

    for (uint32_t local_k = 0; local_k < block_dim; local_k++) {
      reg_c += local_a[block_dim * local_row + local_k] *
               local_b[block_dim * local_k + local_col];
    }

    vx_barrier(0, vx_num_warps());
    vx_fence();
  }

  global_c[dim_n * local_row + local_col] = reg_c;
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  int threads_per_core = vx_num_warps() * vx_num_threads();
  vx_spawn_tasks(threads_per_core, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
