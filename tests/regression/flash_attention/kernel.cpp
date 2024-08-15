#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

// using float_type = float;
using float_type = float16_t;

#define B_ROW BM
#define B_COL BN

inline void thread_block_flashattn(float *S, float *gmem,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock,
                                   const uint32_t threadblock_id_in_cluster,
                                   float *sharedmem_scratchpad,
                                   float *sharedmem_row_max_sum) {
  asm volatile("thread_block_flashattn_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threads_per_threadblock;

  // float ft[8];
  // asm volatile("fmv.s %0, f16" : "=f"(ft[0]));
  // asm volatile("fmv.s %0, f17" : "=f"(ft[1]));
  // asm volatile("fmv.s %0, f18" : "=f"(ft[2]));
  // asm volatile("fmv.s %0, f19" : "=f"(ft[3]));
  // asm volatile("fmv.s %0, f20" : "=f"(ft[4]));
  // asm volatile("fmv.s %0, f21" : "=f"(ft[5]));
  // asm volatile("fmv.s %0, f22" : "=f"(ft[6]));
  // asm volatile("fmv.s %0, f23" : "=f"(ft[7]));

  // row-max
  //
  // one warp handles one row in tile; iterate enough times to cover all the
  // rows
  for (int warp_offset = 0; warp_offset < B_ROW;
       warp_offset += warps_in_threadblock) {
    const uint32_t row = warp_offset + warp_id;
    const uint32_t first_thread_offset = B_COL * row;
    uint32_t thread_offset = first_thread_offset + tid_in_warp;

    float curr_max = S[first_thread_offset];
    constexpr uint32_t load_iter = B_COL / NUM_THREADS;
#pragma GCC unroll
    for (int iter = 0; iter < load_iter; iter++) {
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(curr_max)
                   : "f"(curr_max), "f"(S[thread_offset]));
      thread_offset += NUM_THREADS;
    }
    // get max value across the same-warp threads using smem
    // NOTE: be careful with out-of-bounds
    float *warp_smem = sharedmem_scratchpad + (row * NUM_THREADS);
    warp_smem[tid_in_warp] = curr_max;

    // sync writes to warp_smem
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // 0-th thread collects all other thread's values in the warp
    if (tid_in_warp == 0) {
      for (int iter = 1; iter < NUM_THREADS; iter++) {
        float other = warp_smem[iter];
        asm volatile("fmax.s %0, %1, %2"
                     : "=f"(curr_max)
                     : "f"(curr_max), "f"(other));
      }
      sharedmem_row_max_sum[row] = curr_max;
    }
  }

  // exponential
  //
  // FIXME: placeholder for proper exp
  constexpr uint32_t exp_elem_per_thread = 1;
  // B_ROW / (B_ROW * B_COL / (exp_elem * threads_per_threadblock))
  const uint32_t row_stride =
      (exp_elem_per_thread * threads_per_threadblock) / B_COL;

  asm volatile("thread_block_flashattn_finish_%=:" ::);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

#ifdef RADIANCE
  constexpr uint32_t cores_per_cluster = CORES_PER_CLUSTER;
#else
  constexpr uint32_t cores_per_cluster = 1;
#endif

  uint32_t threads_per_threadblock = (BM * BN) / (ELEM_PER_THREAD);
  const uint32_t hw_threads_per_cluster =
      cores_per_cluster * vx_num_threads() * vx_num_warps();
  // cap maximum threadblock size to # of HW threads in cluster, to prevent
  // multiple "wave" invocations which slows down the kernel
  if (threads_per_threadblock > hw_threads_per_cluster) {
    threads_per_threadblock = hw_threads_per_cluster;
  }
  const uint32_t threadblocks_per_cluster =
      hw_threads_per_cluster / threads_per_threadblock;

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

  uint8_t *smem_S = sharedmem_per_threadblock;
  constexpr uint32_t sharedmem_row_max_sum_size = 2 * sizeof(float) * B_ROW;
  // sharedmem area to store rowmax/rowsum values in softmax
  uint8_t *sharedmem_row_max_sum =
      reinterpret_cast<uint8_t *>(SMEM_ADDR_END) - sharedmem_row_max_sum_size;
  // sharedmem "scratchpad" area to put temporary data, e.g. for tree reduction
  // in rowsum
  // FIXME: size is arbitrary, and out-of bounds is not checked
  constexpr uint32_t sharedmem_scratchpad_size = 0x1000;
  uint8_t *sharedmem_scratchpad =
      sharedmem_row_max_sum - sharedmem_scratchpad_size;

  thread_block_gemm<float_type, /*write_to_gmem=*/true>(
      (const float_type *)arg->addr_a, (const float_type *)arg->addr_b,
      (float *)smem_S /*write result to SMEM */, arg->dim_m, arg->dim_n,
      arg->dim_k, tid_in_threadblock, threads_per_threadblock,
      threadblocks_per_cluster, threadblock_id_in_cluster,
      sharedmem_per_threadblock);

  // protect writes of GEMM results before softmax
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threads_per_threadblock;
  threadblock_barrier(threadblock_id_in_cluster,
                      warps_per_threadblock_per_core);

  thread_block_flashattn(
      (float *)smem_S, (float *)arg->addr_c, tid_in_threadblock,
      threads_per_threadblock, threadblock_id_in_cluster,
      (float *)sharedmem_scratchpad_size, (float *)sharedmem_row_max_sum);
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
