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

constexpr bool DEBUG = true;

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

inline void thread_block_copy_rowmax(const float *src, float *dest,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock,
                                   const uint32_t threadblocks_per_cluster,
                                   const uint32_t threadblock_id_in_cluster) {
  asm volatile("threadblock_copy_rowmax_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  constexpr uint32_t num_warps = B_ROW / NUM_THREADS;
  if (warp_id < num_warps) {
    uint32_t offset = NUM_THREADS * warp_id + tid_in_warp;
    dest[offset] = src[offset];
  }

  threadblock_barrier(threadblock_id_in_cluster,
                      warps_per_threadblock_per_core);

  asm volatile("threadblock_copy_rowmax_finish_%=:" ::);
}

inline void thread_block_copy_tile(const float *src, float *dest,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock,
                                   const uint32_t threadblocks_per_cluster,
                                   const uint32_t threadblock_id_in_cluster) {
  asm volatile("threadblock_copy_tile_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  // FIXME: dedup this pattern
  for (int warp_offset = 0; warp_offset < B_ROW;
       warp_offset += warps_in_threadblock) {
    const uint32_t row = warp_offset + warp_id;
    const uint32_t first_thread_offset = B_COL * row;

    constexpr uint32_t per_row_iter = B_COL / NUM_THREADS;
    uint32_t thread_offset = first_thread_offset + tid_in_warp;
    float per_thread_max = FLT_MIN;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      dest[thread_offset] = src[thread_offset];
      thread_offset += NUM_THREADS;
    }

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);
  }

  asm volatile("threadblock_copy_tile_finish_%=:" ::);
}

template <int order>
inline float exponential_taylor_term(const float x) {
  asm volatile("exponential_taylor_term_start_%=:" ::);

  float res = 1.0f;

  if constexpr (order == 1) {
    res = x;
  } else if constexpr (order == 2) {
    res = x * x;
    res /= 2.0f;
  } else if constexpr (order == 3) {
    res = x * x * x;
    res /= 6.0f;
  }

  asm volatile("exponential_taylor_term_end_%=:" ::);
  return res;
}

__attribute__((always_inline)) inline void thread_block_online_softmax(
    const float *smem_S, float *smem_O, float *smem_P,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock,
    const uint32_t threadblocks_per_cluster,
    const uint32_t threadblock_id_in_cluster, float *smem_scratchpad,
    float *smem_rowmax, float *smem_rowsum) {
  asm volatile("thread_block_flashattn_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  // float ft[8];
  // asm volatile("fmv.s %0, f16" : "=f"(ft[0]));
  // asm volatile("fmv.s %0, f17" : "=f"(ft[1]));
  // asm volatile("fmv.s %0, f18" : "=f"(ft[2]));
  // asm volatile("fmv.s %0, f19" : "=f"(ft[3]));
  // asm volatile("fmv.s %0, f20" : "=f"(ft[4]));
  // asm volatile("fmv.s %0, f21" : "=f"(ft[5]));
  // asm volatile("fmv.s %0, f22" : "=f"(ft[6]));
  // asm volatile("fmv.s %0, f23" : "=f"(ft[7]));

  float *smem_rowmax_this = smem_rowmax + B_ROW;

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
      float prev_rowmax = smem_rowmax[row];
      // stage prev rowmax in scratchpad for warp-wide broadcast
      warp_smem[0] = prev_rowmax;
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(rowmax)
                   : "f"(rowmax), "f"(prev_rowmax));
      smem_rowmax[row] = rowmax;
    }

#endif

    // FIXME: unnecessary?
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // broadcast prev rowmax to all threads in the warp
    // NOTE: memory consistency is a little sketchy here
    const float rowmax_prev = warp_smem[0];
    const float rowmax_this = smem_rowmax_this[row];

    // exponential
    //
    // B_ROW / (B_ROW * B_COL / (exp_elem * threads_per_threadblock))
    // const uint32_t row_stride =
    //     (exp_elem_per_thread * threads_per_threadblock) / B_COL;

    // broadcast updated rowmax to all threads in the warp
    const float rowmax_new = smem_rowmax[row];

    // each thread computes two fp32 elements, downconverts it to fp16, then
    // packs them into one fp32
    constexpr uint32_t elem_per_thread = 1;
    static_assert((B_COL % (elem_per_thread * NUM_THREADS)) == 0,
                  "B_COL condition not met for P compute");

    thread_offset = first_thread_offset + (elem_per_thread * tid_in_warp);
    constexpr uint32_t exp_per_row_iter =
        B_COL / (elem_per_thread * NUM_THREADS);

    asm volatile("flashattn_exp_p_start_%=:" ::);

#pragma GCC unroll
    for (int i = 0; i < exp_per_row_iter; i++) {
      float f0 = smem_S[thread_offset];

      f0 -= rowmax_new;

      // 2nd-order Taylor approximation
      float exp = 1.0f;
      exp += exponential_taylor_term<1>(f0);
      exp += exponential_taylor_term<2>(f0);

      // Store S transposed to the shared memory

      smem_P[thread_offset] = exp;

      thread_offset += NUM_THREADS;
    }

    asm volatile("flashattn_exp_p_end_%=:" ::);


    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // rowsum
    //
    // two-level tree reduction, similar to rowmax

    asm volatile("flashattn_rowsum_start_%=:" ::);

    thread_offset = first_thread_offset + tid_in_warp;
    float per_thread_sum = 0.0f;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      per_thread_sum += smem_P[thread_offset];
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

      const float mi_prev = rowmax_prev;
      // TODO: replace this with a register?
      const float mi_this = rowmax_this;

      const float x = mi_prev - mi_this;
      // 2nd-order Taylor approximation
      float exp = 1.0f;
      exp += exponential_taylor_term<1>(x);
      exp += exponential_taylor_term<2>(x);

      // update rowsum
      const float rowsum_prev = smem_rowsum[row];
      float rowsum_new = exp * rowsum_prev + rowsum;

      smem_rowsum[row] = rowsum_new;
    }

    asm volatile("flashattn_rowsum_end_%=:" ::);

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // Oi rescale
    //
    asm volatile("flashattn_o_rescale_start_%=:" ::);

    thread_offset = first_thread_offset + tid_in_warp;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      float o = smem_O[thread_offset];

      const float mi_prev = rowmax_prev;
      const float mi_new = rowmax_new;

      const float x = mi_prev - mi_new;
      // 2nd-order Taylor approximation
      float exp = 1.0f;
      exp += exponential_taylor_term<1>(x);
      exp += exponential_taylor_term<2>(x);

      // @perf: div vs. expansion on e(-x)?
      o /= exp;

      // update Oi in-place
      smem_O[thread_offset] = o;

      thread_offset += NUM_THREADS;
    }

    asm volatile("flashattn_o_rescale_end_%=:" ::);

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
  constexpr uint32_t smem_Q_size = B_ROW * HEADDIM;
  constexpr uint32_t smem_QK_size = B_ROW * B_COL;
  constexpr uint32_t smem_V_size = B_COL * HEADDIM;
  constexpr uint32_t smem_O_size = B_COL * HEADDIM;
  uint8_t *smem_per_threadblock = reinterpret_cast<uint8_t *>(
      DEV_SMEM_START_ADDR +
      sizeof(float_type) *
          (smem_QK_size + smem_V_size + smem_O_size) *
          threadblock_id_in_cluster);

  float *smem_Q = reinterpret_cast<float *>(smem_per_threadblock);
  float *smem_K = smem_Q + smem_Q_size;
  // in-place multiplication of QK into Q
  float *smem_S = reinterpret_cast<float *>(smem_per_threadblock);
  float *smem_P = smem_S; // in-place update from S to P
  float *smem_V =
      reinterpret_cast<float *>(smem_per_threadblock) + smem_QK_size;
  float *smem_O = reinterpret_cast<float *>(smem_per_threadblock) +
                  smem_QK_size + smem_V_size;

  // allocate rowmax/rowsum storage at the end of the sharedmem address space
  constexpr uint32_t smem_rowmax_size = B_ROW * 3 /* mi, mi~, minew */;
  constexpr uint32_t smem_rowsum_size = B_ROW;
  float *smem_rowmax =
      reinterpret_cast<float *>(SMEM_ADDR_END) - smem_rowmax_size;
  float *smem_rowsum = smem_rowmax - smem_rowsum_size;

  // sharedmem "scratchpad" area to put temporary data, e.g. for tree reduction
  // in rowsum
  // NOTE: out-of bounds is not checked
  // TODO: reduce this from B_ROW to NUM_WARPS
  constexpr uint32_t smem_scratchpad_size =
      B_ROW * NUM_THREADS * 2 /*arbitrary slack*/;
  float *smem_scratchpad = smem_rowsum - smem_scratchpad_size;

  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  // initialize rowmax/rowsum values in sharedmem
  thread_block_init_sharedmem(tid_in_threadblock, threads_per_threadblock,
                              smem_O, smem_rowmax, smem_rowsum);

  const float *gmem_Q = reinterpret_cast<float *>(arg->addr_q);
  const float *gmem_K = reinterpret_cast<float *>(arg->addr_k);
  const float *gmem_V = reinterpret_cast<float *>(arg->addr_v);
  float *gmem_O = reinterpret_cast<float *>(arg->addr_o);

  float *gmem_tmp_d0 = reinterpret_cast<float *>(0xd0000000UL);
  float *gmem_tmp_d1 = reinterpret_cast<float *>(0xd1000000UL);
  float *gmem_tmp_d2 = reinterpret_cast<float *>(0xd2000000UL);
  float *gmem_tmp_d3 = reinterpret_cast<float *>(0xd3000000UL);
  float *gmem_tmp_d4 = reinterpret_cast<float *>(0xd4000000UL);
  float *gmem_tmp_d5 = reinterpret_cast<float *>(0xd5000000UL);
  float *gmem_tmp_e0 = reinterpret_cast<float *>(0xe0000000UL);
  float *gmem_tmp_e1 = reinterpret_cast<float *>(0xe1000000UL);
  float *gmem_tmp_e2 = reinterpret_cast<float *>(0xe2000000UL);
  float *gmem_tmp_e3 = reinterpret_cast<float *>(0xe3000000UL);

  // "inner loop" along the columns of K^T
  const uint32_t k_tiles = (dim_seqlen / B_COL);
  for (uint32_t tile_k = 0; tile_k < k_tiles; tile_k++) {

    const float *tile_S = nullptr;

    constexpr bool skip_gemm_qk = true;
    if constexpr (!skip_gemm_qk) {
      // clear out accumulators
      initialize_accum_regs<0>();
      initialize_accum_regs<1>();

      static_assert(B_ROW == B_COL, "currently only supports square tiles");

      // load Q
      load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_ROW,
                        HEADDIM>(
          dim_seqlen, 0 /*FIXME: only work on first B_ROW rows of Q for now*/,
          0 /* always 0 because dim_k == headdim */, gmem_Q, smem_Q,
          tid_in_threadblock);

      // load K
      load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_COL,
                        HEADDIM>(dim_seqlen, tile_k,
                                 0 /* always 0 because dim_k == headdim */,
                                 gmem_K, smem_K, tid_in_threadblock);

      // GMEM->SMEM and compute barrier
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);

      // GEMM I: S = Q*K
      thread_block_gemm_single_tile<float, MemLayout::MN_major,
                                    MemLayout::MN_major,
                                    /*load_accum=*/false,
                                    /*write_to_smem=*/true>(
          smem_Q, smem_K, nullptr /*ignore accum*/, smem_S, tid_in_threadblock,
          threads_per_threadblock, threadblocks_per_cluster,
          threadblock_id_in_cluster);

      // tile_S = smem_S;
    } else {
      // load Q*K
      load_tile_to_smem<float, MemLayout::K_major, MemLayout::K_major, B_COL,
                        HEADDIM>(dim_seqlen, 0, tile_k, gmem_Q /*=gmem_S*/,
                                 smem_S, tid_in_threadblock);
      // the above should be equivalent to:
      // load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_COL,
      //                   HEADDIM>(dim_seqlen, tile_k, 0, gmem_Q /*=gmem_S*/,
      //                            smem_S, tid_in_threadblock);

      // tile_S = reinterpret_cast<float *>(arg->addr_q);
    }

    // protect GEMM result writes (smem_S) before softmax
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    thread_block_online_softmax(
        smem_S, smem_O, smem_P, tid_in_threadblock, threads_per_threadblock,
        threadblocks_per_cluster, threadblock_id_in_cluster, smem_scratchpad,
        smem_rowmax, smem_rowsum);

    // FIXME unnecessary?
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    if constexpr (DEBUG) {
      if (tile_k == 0) {
        thread_block_copy_tile(
            smem_P, gmem_tmp_d0, tid_in_threadblock, threads_per_threadblock,
            threadblocks_per_cluster, threadblock_id_in_cluster);
        thread_block_copy_tile(
            smem_O, gmem_tmp_d2, tid_in_threadblock, threads_per_threadblock,
            threadblocks_per_cluster, threadblock_id_in_cluster);
        thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e0, tid_in_threadblock,
                                 threads_per_threadblock,
                                 threadblocks_per_cluster,
                                 threadblock_id_in_cluster);
        thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e2, tid_in_threadblock,
                                 threads_per_threadblock,
                                 threadblocks_per_cluster,
                                 threadblock_id_in_cluster);
      } else if (tile_k == k_tiles - 1) {
        thread_block_copy_tile(
            smem_P, gmem_tmp_d1, tid_in_threadblock, threads_per_threadblock,
            threadblocks_per_cluster, threadblock_id_in_cluster);
        thread_block_copy_tile(
            smem_O, gmem_tmp_d3, tid_in_threadblock, threads_per_threadblock,
            threadblocks_per_cluster, threadblock_id_in_cluster);
        thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e1, tid_in_threadblock,
                                 threads_per_threadblock,
                                 threadblocks_per_cluster,
                                 threadblock_id_in_cluster);
        thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e3, tid_in_threadblock,
                                 threads_per_threadblock,
                                 threadblocks_per_cluster,
                                 threadblock_id_in_cluster);
      }

      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }

    // GEMM II: O = O + P*V

    // clear out accumulators
    initialize_accum_regs<0>();
    initialize_accum_regs<1>();

    // V dimension is [seqlen, headdim], stored N(headdim)-major
    load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_COL,
                      HEADDIM>(
        HEADDIM, 0 /* 0 because always reads the full N-dimension */, tile_k,
        gmem_V, smem_V, tid_in_threadblock);

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // FIXME: support MN_major for A for ideal performance
    thread_block_gemm_single_tile<float, MemLayout::K_major,
                                  MemLayout::MN_major,
                                  /*load_accum=*/true,
                                  /*write_to_smem=*/true>(
        smem_P, smem_V, smem_O /*load accum*/, smem_O,
        tid_in_threadblock, threads_per_threadblock, threadblocks_per_cluster,
        threadblock_id_in_cluster);

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    if constexpr (DEBUG) {
      if (tile_k == 0) {
        thread_block_copy_tile(
            smem_O, gmem_tmp_d4, tid_in_threadblock, threads_per_threadblock,
            threadblocks_per_cluster, threadblock_id_in_cluster);
      } else if (tile_k == k_tiles - 1) {
        thread_block_copy_tile(
            smem_O, gmem_tmp_d5, tid_in_threadblock, threads_per_threadblock,
            threadblocks_per_cluster, threadblock_id_in_cluster);
      }

      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }
  }
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
