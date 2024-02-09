#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	uint32_t num_points = arg->num_points;
	float* src_ptr = (float*)arg->src_addr;
	float* dst_ptr = (float*)arg->dst_addr;

	dst_ptr[task_id] = 2 * src_ptr[task_id];
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
