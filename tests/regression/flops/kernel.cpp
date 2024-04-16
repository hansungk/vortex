#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  const float *A = (const float *)arg->addr_src;
  float *C = (float *)arg->addr_dst;

  int incr = A[task_id];
  float sum = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  float sum4 = 0.0f;
  float sum5 = 0.0f;
#pragma unroll 8
  for (int i = 0; i < 5000; i++) {
    sum1 = sum2 + 5.0f;
    sum2 = sum3 + 5.0f;
    sum3 = sum4 + 5.0f;
    sum4 = sum5 + 5.0f;
    sum5 = sum1 + 5.0f;
  }

  sum = sum1 + sum2 + sum3 + sum4 + sum5;
  C[task_id] = static_cast<float>(sum);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  const uint32_t grid_size = arg->size;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
