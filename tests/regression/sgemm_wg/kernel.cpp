#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const float *global_a = (const float *)arg->addr_a;
  float *global_c = (float *)arg->addr_c;

  // assumes NT == NW == matrix_dim
  const uint32_t dim = arg->matrix_dim;
  const uint32_t row = vx_warp_id();
  const uint32_t col = vx_thread_id();

  float *local_c = (float *)DEV_SMEM_START_ADDR;
  float *local_a = (float *)DEV_SMEM_START_ADDR + (dim * dim);
  float *local_b = (float *)DEV_SMEM_START_ADDR + 2 * (dim * dim);

  local_a[dim * row + col] = global_a[dim * row + col];
  local_c[dim * row + col] = 0.0f;

  vx_barrier(0, vx_num_warps());

  for (uint32_t k = 0; k < dim; k++) {
    local_c[dim * row + col] += local_a[dim * row + k] * local_a[dim * k + col];
  }

  vx_barrier(0, vx_num_warps());

  global_c[dim * row + col] = local_c[dim * row + col];
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
        int threads_per_core = vx_num_warps() * vx_num_threads();
	vx_spawn_tasks(threads_per_core, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
