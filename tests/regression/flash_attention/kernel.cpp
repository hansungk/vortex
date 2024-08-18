#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include <float.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define B_ROW BM
#define B_COL BN
// FIXME
#define HEADDIM B_COL

inline void thread_block_init_sharedmem(const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock,
                                   float *smem_O,
                                   float *smem_rowmax,
                                   float *smem_rowsum) {
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;

  static_assert((B_ROW % NUM_THREADS) == 0,
                "B_ROW must be a multiple of NUM_THREADS");
  // FIXME: this shouldn't be necessary
  static_assert(B_ROW < (NUM_THREADS * CORES_PER_CLUSTER * NUM_WARPS),
                "not enough warps to initialize rowmax/rowsum");

  constexpr uint32_t num_warps = B_ROW / NUM_THREADS;
  if (warp_id < num_warps) {
    uint32_t offset = NUM_THREADS * warp_id + tid_in_warp;
    // mi, mi~, minew
    smem_rowmax[offset] = FLT_MIN;
    smem_rowmax[offset + B_ROW] = FLT_MIN;
    smem_rowmax[offset + 2 * B_ROW] = FLT_MIN;
    smem_rowsum[offset] = 0.0f;
  }

  // FIXME: dedup this pattern
  for (int warp_offset = 0; warp_offset < B_COL;
       warp_offset += warps_in_threadblock) {
    // each warp clears out a row of smem_O
    const uint32_t row = warp_offset + warp_id;
    uint32_t thread_offset = HEADDIM * row + tid_in_warp;
    constexpr uint32_t per_row_iter = HEADDIM / NUM_THREADS;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      smem_O[thread_offset] = 0.0f;
      thread_offset += NUM_THREADS;
    }
  }
}

inline void thread_block_online_softmax(
    float *smem_S, float *smem_O, float *smem_P,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock,
    const uint32_t threadblock_id_in_cluster, float *smem_scratchpad,
    float *smem_rowmax, float *smem_rowsum) {
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

  volatile float *gmem_tmp0 = reinterpret_cast<volatile float *>(0xd0000000UL);
  volatile float *gmem_tmp1 = reinterpret_cast<volatile float *>(0xe0000000UL);
  volatile float *gmem_tmp2 = reinterpret_cast<volatile float *>(0xf0000000UL);

  float *smem_rowmax_prev = smem_rowmax;
  float *smem_rowmax_new = smem_rowmax + B_ROW;
  float *smem_rowmax_this = smem_rowmax + 2 * B_ROW;

  for (int warp_offset = 0; warp_offset < B_ROW;
       warp_offset += warps_in_threadblock) {
    const uint32_t row = warp_offset + warp_id;
    const uint32_t first_thread_offset = B_COL * row;

    // rowmax
    //
    // two-level tree reduction: reduce each row into NUM_THREADS intermediate
    // maxes, then reduce it to one global max
    // one warp handles one row in tile

// #define DUMB_ROWMAX
#ifdef DUMB_ROWMAX
    if (tid_in_warp == 0) {
      float max = S[first_thread_offset];
#pragma GCC unroll
      for (int i = 0; i < B_COL; i++) {
        asm volatile("fmax.s %0, %1, %2"
                     : "=f"(max)
                     : "f"(max), "f"(S[first_thread_offset + i]));
      }
      smem_rowmax[row] = max;
      gmem_tmp0[row] = max;
    }

#else
    static_assert((B_COL % NUM_THREADS) == 0,
                  "B_COL must be a multiple of NUM_THREADS");
    constexpr uint32_t per_row_iter = B_COL / NUM_THREADS;
    uint32_t thread_offset = first_thread_offset + tid_in_warp;
    float per_thread_max = FLT_MIN;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const float next = smem_S[thread_offset];
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(per_thread_max)
                   : "f"(per_thread_max), "f"(next));
      thread_offset += NUM_THREADS;
    }
    // stage per-thread max value in smem
    // FIXME: threadblock_id needs to be in here too
    float *warp_smem = smem_scratchpad + (warp_id * NUM_THREADS);
    warp_smem[tid_in_warp] = per_thread_max;

    // sync writes to warp_smem
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // elect 0-th thread to reduce all other thread's values in the warp
    if (tid_in_warp == 0) {
      float rowmax = per_thread_max;
      for (int iter = 1; iter < NUM_THREADS; iter++) {
        float other = warp_smem[iter];
        asm volatile("fmax.s %0, %1, %2"
                     : "=f"(rowmax)
                     : "f"(rowmax), "f"(other));
      }
      smem_rowmax_this[row] = rowmax;

      // update previous rowmax
      // i.e. mi_new = max(mi, mij)
      float prev_rowmax = smem_rowmax_prev[row];
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(rowmax)
                   : "f"(rowmax), "f"(prev_rowmax));
      smem_rowmax_new[row] = rowmax;
      gmem_tmp0[row] = rowmax;
    }
#endif

    // FIXME: unnecessary?
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // exponential
    //
    // B_ROW / (B_ROW * B_COL / (exp_elem * threads_per_threadblock))
    // const uint32_t row_stride =
    //     (exp_elem_per_thread * threads_per_threadblock) / B_COL;

    // broadcast rowmax to all threads in the warp
    const float rowmax_new = smem_rowmax_new[row];

    // each thread computes two fp32 elements, downconverts it to fp16, then
    // packs them into one fp32
    constexpr uint32_t elem_per_thread = 1;
    static_assert((B_COL % (elem_per_thread * NUM_THREADS)) == 0,
                  "B_COL condition not met for P compute");

    thread_offset = first_thread_offset + (elem_per_thread * tid_in_warp);
    constexpr uint32_t exp_per_row_iter =
        B_COL / (elem_per_thread * NUM_THREADS);
#pragma GCC unroll
    for (int i = 0; i < exp_per_row_iter; i++) {
      float f0 = smem_S[thread_offset];
      // float f1 = S[thread_offset + 1];

      // FIXME: placeholder for proper exp
      f0 -= rowmax_new;
      // f1 -= rowmax_new;
      // float16_t h0 = NN_float_to_half(f0);
      // float16_t h1 = NN_float_to_half(f1);

      // Store S transposed to the shared memory

      smem_S[thread_offset] = f0;
      // S[thread_offset + 1] = f1;
      gmem_tmp1[thread_offset] = f0;

      thread_offset += NUM_THREADS;
    }

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // rowsum
    //
    // two-level tree reduction, similar to rowmax

    thread_offset = first_thread_offset + tid_in_warp;
    float per_thread_sum = 0.0f;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      per_thread_sum += smem_S[thread_offset];
      thread_offset += NUM_THREADS;
    }
    // stage per-thread sum value in smem
    // FIXME: threadblock_id needs to be in here too
    warp_smem = smem_scratchpad + (warp_id * NUM_THREADS);
    warp_smem[tid_in_warp] = per_thread_sum;

    // sync writes to warp_smem
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // 0-th thread collects all other thread's values in the warp
    if (tid_in_warp == 0) {
      float rowsum = per_thread_sum;
      for (int iter = 1; iter < NUM_THREADS; iter++) {
        float other = warp_smem[iter];
        rowsum += other;
      }

      const float mi_prev = smem_rowmax_prev[row];
      const float mi_this = smem_rowmax_this[row];
      const float exp = mi_prev - mi_this;

      // update rowsum
      const float rowsum_prev = smem_rowsum[row];
      // FIXME: placeholder for exponential
      float rowsum_new = exp * rowsum_prev + rowsum;
      smem_rowsum[row] = rowsum_new;
    }

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // Oi rescale
    //
    thread_offset = first_thread_offset + tid_in_warp;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      float fval = smem_O[thread_offset];

      const float mi_prev = smem_rowmax_prev[row];
      const float mi_new = smem_rowmax_new[row];
      const float exp = mi_prev - mi_new;

      // FIXME: placeholder for proper exp
      fval *= exp;

      // update Oi in-place
      smem_O[thread_offset] = fval;

      thread_offset += NUM_THREADS;
    }

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

  }

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

  // FIXME: headdim not considered
  uint32_t threads_per_threadblock = (B_ROW * B_COL) / (ELEM_PER_THREAD);
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

  const uint32_t dim_seqlen = arg->dim_seqlen;
  const uint32_t dim_headdim = arg->dim_headdim;

  // "static" shared memory allocation.  This would determine maximum
  // threadblock occupancy in a cluster
  const uint32_t smem_QK_size = B_ROW * B_COL;
  const uint32_t smem_V_size = B_COL * HEADDIM;
  const uint32_t smem_O_size = B_COL * HEADDIM;
  uint8_t *smem_per_threadblock = reinterpret_cast<uint8_t *>(
      DEV_SMEM_START_ADDR +
      sizeof(float_type) *
          (smem_QK_size + smem_V_size + smem_O_size) *
          threadblock_id_in_cluster);

  uint8_t *smem_S = smem_per_threadblock;
  uint8_t *smem_P = smem_S; // in-place update from S to P
  uint8_t *smem_V = smem_per_threadblock + sizeof(float) * smem_QK_size;
  uint8_t *smem_O =
      smem_per_threadblock + sizeof(float) * (smem_QK_size + smem_V_size);

  // allocate rowmax/rowsum storage at the end of the sharedmem address space
  constexpr uint32_t smem_rowmax_size = sizeof(float) * B_ROW * 3 /* mi, mi~, minew */;
  constexpr uint32_t smem_rowsum_size = sizeof(float) * B_ROW;
  uint8_t *smem_rowmax =
      reinterpret_cast<uint8_t *>(SMEM_ADDR_END) - smem_rowmax_size;
  uint8_t *smem_rowsum = smem_rowmax - smem_rowsum_size;

  // sharedmem "scratchpad" area to put temporary data, e.g. for tree reduction
  // in rowsum
  // NOTE: out-of bounds is not checked
  constexpr uint32_t smem_scratchpad_size =
      sizeof(float) * B_ROW * NUM_THREADS * 2 /*arbitrary slack*/;
  uint8_t *smem_scratchpad =
      smem_rowmax - smem_scratchpad_size;

  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threads_per_threadblock;

  // initialize rowmax/rowsum values in sharedmem
  thread_block_init_sharedmem(tid_in_threadblock, threads_per_threadblock,
                              (float *)smem_O,
                              (float *)smem_rowmax,
                              (float *)smem_rowsum);

#define SKIP_GEMM
#ifndef SKIP_GEMM
  thread_block_gemm<float_type, /*write_to_gmem=*/true>(
      (const float_type *)arg->addr_q, (const float_type *)arg->addr_k,
      (float *)smem_S /*write result to SMEM */, arg->dim_m, arg->dim_n,
      arg->dim_k, tid_in_threadblock, threads_per_threadblock,
      threadblocks_per_cluster, threadblock_id_in_cluster,
      smem_per_threadblock);

  // protect writes of GEMM results before softmax
  threadblock_barrier(threadblock_id_in_cluster,
                      warps_per_threadblock_per_core);

  float *tile_S = (float *)smem_S;
#else
  float *tile_S = (float *)arg->addr_q;
#endif

  thread_block_online_softmax(
      tile_S, (float *)smem_O, (float *)smem_P, tid_in_threadblock,
      threads_per_threadblock, threadblock_id_in_cluster,
      (float *)smem_scratchpad, (float *)smem_rowmax, (float *)smem_rowsum);

  // FIXME unnecessary?
  threadblock_barrier(threadblock_id_in_cluster,
                      warps_per_threadblock_per_core);

  thread_block_gemm_single_tile(smem_P, smem_V, tid_in_threadblock,
                                threads_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  // FIXME:: use actuall seqlen/headdim
  const uint32_t problem_size = (B_ROW * B_COL) / (ELEM_PER_THREAD);
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
