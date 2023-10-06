#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
	uint32_t num_points = arg->num_points;
	uint32_t points_per_core = num_points / vx_num_warps();
	int tid = vx_thread_lid();
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;

	int32_t ref_value = src_ptr[task_id];
	int ref_negative = ref_value < 0;
	vx_split(ref_negative);
	if (ref_negative) {
		ref_value = 0;
	}
	vx_join();
	
	dst_ptr[task_id] = ref_value;
}

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	int num_warps = vx_num_warps();
	vx_spawn_tasks(arg->num_points, (vx_spawn_tasks_cb)kernel_body, arg);
}