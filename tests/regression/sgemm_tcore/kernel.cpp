#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

#ifdef RADIANCE
  constexpr uint32_t cores_per_cluster = CORES_PER_CLUSTER;
#else
  constexpr uint32_t cores_per_cluster = 1;
#endif

  constexpr uint32_t threads_per_threadblock_theoretical =
      (BM * BN) / (ELEM_PER_THREAD);
  constexpr uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * NUM_THREADS * NUM_WARPS;
  // cap maximum threadblock size to # of HW threads in cluster, to prevent
  // multiple "wave" invocations which slows down the kernel
  constexpr uint32_t threads_per_threadblock =
      (threads_per_threadblock_theoretical > hw_threads_per_cluster)
          ? hw_threads_per_cluster
          : threads_per_threadblock_theoretical;
  constexpr uint32_t threadblocks_per_cluster =
      hw_threads_per_cluster / threads_per_threadblock;
  constexpr uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_cluster =
      threadblock_id % threadblocks_per_cluster;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;
  const uint32_t problem_size = (dim_m * dim_n) / (ELEM_PER_THREAD);
  const uint32_t num_threadblocks = problem_size / threads_per_threadblock;

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  uint8_t *sharedmem_per_threadblock = reinterpret_cast<uint8_t *>(
      DEV_SMEM_START_ADDR + sizeof(float_type) * 2 /*overkill for non-dma*/ *
                                (2 * BM * BK) * threadblock_id_in_cluster);

  thread_block_gemm<float_type, threads_per_threadblock,
                    /*write_to_gmem=*/true>(
      (const float_type *)arg->addr_a, (const float_type *)arg->addr_b,
      (float *)arg->addr_c, arg->dim_m, arg->dim_n, arg->dim_k,
      tid_in_threadblock, threadblocks_per_cluster, threadblock_id_in_cluster,
      sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t problem_size = (arg->dim_m * arg->dim_n) / (ELEM_PER_THREAD);
  const uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps();
  // prevent launching more threads than the necessary problem size
  // TODO: this does not take into account multiple clusters
  const uint32_t grid_size = (problem_size > hw_threads_per_cluster)
                                 ? hw_threads_per_cluster
                                 : problem_size;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
