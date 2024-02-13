#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  uint32_t num_points = arg->num_points;
  float *src_ptr = (float *)arg->src_addr;
  float *dst_ptr = (float *)arg->dst_addr;

  float *local_a = (float *)DEV_SMEM_START_ADDR;

  local_a[num_points - 1 - task_id] = 2 * src_ptr[num_points - 1 - task_id];
  // local_a[task_id] = 2 * src_ptr[task_id];

  vx_barrier(0, vx_num_warps());

  dst_ptr[task_id] = local_a[task_id];
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
        int threads_per_core = vx_num_warps() * vx_num_threads();
	vx_spawn_tasks(threads_per_core, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
