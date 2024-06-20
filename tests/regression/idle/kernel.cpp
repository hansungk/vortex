#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define NUM_CLUSTERS 1
#define NUM_THREADS_IN_CLUSTER 256

#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  constexpr uint32_t timer = 50000;
  uint32_t counter = 0;
  while ((counter++) < timer) {
    asm("");
  }
  // to prevent optimize-out
  // reinterpret_cast<uint32_t *>(arg->addr_c)[0] = counter;

  vx_tmc(0);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  // const uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;
  // const uint32_t grid_size = num_threads_in_cluster * NUM_CLUSTERS;
  const uint32_t grid_size = 1;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
