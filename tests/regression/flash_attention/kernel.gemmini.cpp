#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"
#include "flash_impl.hpp"

constexpr bool DEBUG = true;

static_assert(GEMMINI_DMA && !WARP_SPECIALIZED,
              "GEMMINI_DMA should be set and WARP_SPECIALIZED unset");

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

#ifdef RADIANCE
  constexpr uint32_t cores_per_cluster = CORES_PER_CLUSTER;
#else
  constexpr uint32_t cores_per_cluster = 1;
#endif

  // FIXME: headdim not considered
  constexpr uint32_t threads_per_threadblock_theoretical =
      (B_ROW * B_COL) / (ELEM_PER_THREAD);
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

  const uint32_t threadblock_id = task_id / threads_per_threadblock;
  const uint32_t threadblock_id_in_cluster =
      threadblock_id % threadblocks_per_cluster;
  const uint32_t tid_in_threadblock = task_id % threads_per_threadblock;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  constexpr uint32_t warps_in_threadblock =
      threads_per_threadblock / NUM_THREADS;

  // warpgroup context
  constexpr uint32_t threads_per_warpgroup =
      threads_per_threadblock / (WARP_SPECIALIZED ? 2 : 1);
  constexpr uint32_t warpgroups_per_cluster =
      threadblocks_per_cluster * (WARP_SPECIALIZED ? 2 : 1);
  const uint32_t warps_per_warpgroup_per_core =
      NUM_WARPS / warpgroups_per_cluster;
  const uint32_t warpgroup_id = task_id / threads_per_warpgroup;
  const uint32_t warpgroup_id_in_cluster =
      warpgroup_id % warpgroups_per_cluster;
  const uint32_t tid_in_warpgroup = tid_in_threadblock % threads_per_warpgroup;

  const uint32_t dim_seqlen = arg->dim_seqlen;
  const uint32_t dim_headdim = arg->dim_headdim;

  // get global memory addresses from kernel arguments
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
  float *gmem_tmp_d6 = reinterpret_cast<float *>(0xd6000000UL);
  float *gmem_tmp_d7 = reinterpret_cast<float *>(0xd7000000UL);
  float *gmem_tmp_e0 = reinterpret_cast<float *>(0xe0000000UL);
  float *gmem_tmp_e1 = reinterpret_cast<float *>(0xe1000000UL);
  float *gmem_tmp_e2 = reinterpret_cast<float *>(0xe2000000UL);
  float *gmem_tmp_e3 = reinterpret_cast<float *>(0xe3000000UL);

  // static shared memory allocation
  constexpr uint32_t smem_Q_size = B_ROW * HEADDIM;
  constexpr uint32_t smem_K_size = B_COL * HEADDIM;
  constexpr uint32_t smem_QK_size = B_ROW * B_COL;
  constexpr uint32_t smem_V_size = B_COL * HEADDIM;
  constexpr uint32_t smem_O_size = B_COL * HEADDIM;
  static_assert(
      threads_per_threadblock == NUM_WARPS * NUM_THREADS * CORES_PER_CLUSTER,
      "flashattention kernel assumes 1 threadblock occupancy per cluster");
  uint8_t *smem_per_threadblock = reinterpret_cast<uint8_t *>(
      DEV_SMEM_START_ADDR);
  float *smem_cursor = reinterpret_cast<float *>(smem_per_threadblock);
  // float *smem_cursor = reinterpret_cast<float *>(DEV_FAKE_SMEM_START_ADDR);
  float *smem_Q0 = smem_cursor;
  smem_cursor += smem_Q_size;
  float *smem_Q1 = smem_cursor;
  smem_cursor += smem_Q_size;
  float *smem_K0 = smem_cursor;
  smem_cursor += smem_K_size;
  float *smem_K1 = smem_cursor;
  smem_cursor += smem_K_size;
  float *smem_V0 = smem_cursor;
  smem_cursor += smem_V_size;
  float *smem_V1 = smem_cursor;
  smem_cursor += smem_V_size;
  float *smem_S0 = smem_cursor;
  smem_cursor += smem_QK_size;
  float *smem_S1 = smem_cursor;
  smem_cursor += smem_QK_size;
  float *smem_P0 = smem_cursor;
  smem_cursor += smem_QK_size;
  float *smem_P1 = smem_cursor;
  smem_cursor += smem_QK_size;
  float *smem_O0 = smem_cursor;
  smem_cursor += smem_O_size;
  float *smem_O1 = smem_cursor;
  smem_cursor += smem_O_size;

  // NOTE: this has to match with smem_*
  static_assert(sizeof(elem_t) == sizeof(float));
  constexpr uint32_t spad_addr_factor = DIM * sizeof(elem_t);
  constexpr uint32_t spad_addr_Q0 = 0;
  constexpr uint32_t spad_addr_Q1 =
      spad_addr_Q0 + (smem_Q_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_K0 =
      spad_addr_Q1 + (smem_Q_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_K1 =
      spad_addr_K0 + (smem_K_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_V0 =
      spad_addr_K1 + (smem_K_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_V1 =
      spad_addr_V0 + (smem_V_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_S0 =
      spad_addr_V1 + (smem_V_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_S1 =
      spad_addr_S0 + (smem_QK_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_P0 =
      spad_addr_S1 + (smem_QK_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_P1 =
      spad_addr_P0 + (smem_QK_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_O0 =
      spad_addr_P1 + (smem_QK_size * sizeof(float) / spad_addr_factor);
  constexpr uint32_t spad_addr_O1 =
      spad_addr_O0 + (smem_O_size * sizeof(float) / spad_addr_factor);

  // allocate rowmax/rowsum storage at the end of the sharedmem address space
  constexpr uint32_t smem_rowmax_size = B_ROW * ROWMAX_SETS;
  constexpr uint32_t smem_rowsum_size = B_ROW;
  constexpr uint32_t smem_O_row_scale_size = B_ROW;
  // FIXME: dangerous
  smem_cursor = reinterpret_cast<float *>(0xff038000);

  float *smem_rowmax_0 = smem_cursor;
  smem_cursor += smem_rowmax_size;
  float *smem_rowmax_1 = smem_cursor;
  smem_cursor += smem_rowmax_size;
  float *smem_rowsum_0 = smem_cursor;
  smem_cursor += smem_rowsum_size;
  float *smem_rowsum_1 = smem_cursor;
  smem_cursor += smem_rowsum_size;
  float *smem_O_row_scale_0 = smem_cursor;
  smem_cursor += smem_O_row_scale_size;
  float *smem_O_row_scale_1 = smem_cursor;
  smem_cursor += smem_O_row_scale_size;

  // sharedmem "scratchpad" area to put temporary data, e.g. for tree reduction
  // in rowsum
  // NOTE: out-of bounds is not checked
  constexpr uint32_t smem_scratchpad_size =
      threads_per_warpgroup * 2 /*arbitrary slack*/;
  float *smem_scratchpad_0 = smem_cursor;
  smem_cursor += smem_scratchpad_size;
  float *smem_scratchpad_1 = smem_cursor;
  smem_cursor += smem_scratchpad_size;

  // initialize rowmax/rowsum values in sharedmem
  thread_block_init_sharedmem(tid_in_warpgroup, threads_per_warpgroup, smem_O0,
                              smem_rowmax_0, smem_rowsum_0, smem_O_row_scale_0);
  thread_block_init_sharedmem(tid_in_warpgroup, threads_per_warpgroup, smem_O1,
                              smem_rowmax_1, smem_rowsum_1, smem_O_row_scale_1);

  constexpr uint32_t global_barrier_id = NUM_WARPS - 1; // arbitrary

  // // delay warpgroup 0 by 1 iteration to do ping-pong scheduling
  // if (WARP_SPECIALIZED && warpgroup_id == 1) {
  //   threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);
  // }

  static_assert(!GEMMINI_DMA || Q_IS_K_MAJOR,
                "DMA code assumes Q matrix is stored K-major");

  // skip everything except DMA in the loop FSM
  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);
  constexpr uint32_t skips_only_a =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);
  constexpr uint32_t skips_mvout_spad =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/0);
  constexpr uint32_t skips_matmul =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/1);

  if constexpr (GEMMINI_DMA) {
    if (tid_in_warpgroup == 0) {
      gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);

      // configure DMA for the full Q matrix
      gemmini_extended3_config_ld(HEADDIM * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                  false, 0);
      // configure DMA for the full K matrix
      gemmini_extended3_config_ld(dim_seqlen * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                  false, 1);
      // configure DMA for Q*K store
      gemmini_extended_config_st(B_COL * sizeof(elem_t), 0,
                                 MVIN_SCALE_IDENTITY);
      gemmini_fence();
    }
  }

  // NOTE about barriers: Placing barriers around thread-divergent branches may
  // cause bugs, because the Vortex core doesn't check for tmask for barriers.
  // The compiler might decide to duplicate vx_bar into both paths of a
  // conditional branch, which will get evaluated twice because of the way
  // branches are handled in SIMT; this might result in stalls especially when
  // other warps behave differently on the branch condition.
  // threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

  // move Q and K into SMEM before the loop starts
  //
  static_assert(B_ROW == B_COL, "currently only supports square tiles");

  asm volatile("dma_move_start_%=:" ::);

  if (tid_in_warpgroup == 0) {
    // make sure to read from the correct row of Q
    const float *gmem_Q_tile = gmem_Q + HEADDIM * B_ROW * warpgroup_id;
    const float *gmem_K_tile = gmem_K;
    // configure the GMEM addresses for the DMA to read from
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_Q_tile),
                             (uint64_t)(gmem_K_tile), k_LOOP_WS_CONFIG_ADDRS_AB)
    // configure address strides for the DMA
    GEMMINI_CISC_CMD_R((dim_seqlen << 20) | (HEADDIM << 8) |
                       8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
    gemmini_fence();

#define GEMMINI_DMA_CISC
#ifdef GEMMINI_DMA_CISC
    // the target addresses of this should match with spad_addr_Q0 and
    // spad_addr_K0 set in this kernel
    GEMMINI_CISC_CMD_I(10);
    gemmini_fence();
#else
    // do DMA
    //
    // among other things, this also configures CONFIG_BOUNDS so that the
    // DMA knows the full matrix dimensions
    sp_tiled_matmul_full_spad_ws(
        spad_addr_Q, spad_addr_K,
        /*spad_D=*/0, /*spad_C=*/spad_addr_S,
        /*I=*/(B_ROW / DIM), /*J=*/(B_COL / DIM), /*K=*/(HEADDIM / DIM),
        /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
        /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
        /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
    gemmini_fence();
#endif

    // re-configure DMA for K and V load that will later happen in the loop
    // GMEM addr stride for K
    gemmini_extended3_config_ld(dim_seqlen * sizeof(elem_t),
                                MVIN_SCALE_IDENTITY, false, 0);
    // GMEM addr stride for V
    gemmini_extended3_config_ld(HEADDIM * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 1);
    gemmini_fence();
  }

  asm volatile("dma_move_end_%=:" ::);

  // protect write to SMEM
  threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

  // if constexpr (DEBUG) {
  //   thread_block_copy_tile<B_ROW, HEADDIM>(smem_Q0, gmem_tmp_d0, tid_in_warpgroup,
  //                          threads_per_warpgroup, warpgroup_id_in_cluster);
  //   thread_block_copy_tile<HEADDIM, B_COL>(smem_K0, gmem_tmp_d1, tid_in_warpgroup,
  //                          threads_per_warpgroup, warpgroup_id_in_cluster);
  //   threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);
  // }

  asm volatile ("tile_loop_start_%=:" :: );

  // "inner loop" along the columns of K^T
  const uint32_t k_tiles = (dim_seqlen / B_COL);
  for (uint32_t tile_k = 0; tile_k < k_tiles + 2 /*pipeline latency*/;
       tile_k++) {
    if constexpr (DEBUG) {
      // barrier for debugging
      threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);
    }

    // select the correct double buffer by tile iteration
    // all iterations work on the same Q row tile; no ping-pong necessary
    asm volatile ("dbuf_sel_start_%=:" :: );
    // FIXME speedup by doing arithmetic
    float *smem_Q = smem_Q0;
    float *smem_K_consume = (tile_k & 1) ? smem_K1 : smem_K0;
    float *smem_K_produce = (tile_k & 1) ? smem_K0 : smem_K1;
    float *smem_V_consume = (tile_k & 1) ? smem_V1 : smem_V0;
    float *smem_V_produce = (tile_k & 1) ? smem_V0 : smem_V1;
    float *smem_S_consume = (tile_k & 1) ? smem_S1 : smem_S0;
    float *smem_S_produce = (tile_k & 1) ? smem_S0 : smem_S1;
    float *smem_P_consume = (tile_k & 1) ? smem_P1 : smem_P0;
    float *smem_P_produce = (tile_k & 1) ? smem_P0 : smem_P1;
    // O tile is sequentially updated at every iteration; no ping-pong
    // necessary
    float *smem_O = smem_O0;
    // FIXME: O_row_scale/rowmax/rowsum/spad shouldn't really need ping-pong
    float *smem_O_row_scale =
        (tile_k & 1) ? smem_O_row_scale_1 : smem_O_row_scale_0;
    float *smem_rowmax = (tile_k & 1) ? smem_rowmax_1 : smem_rowmax_0;
    float *smem_rowsum = (tile_k & 1) ? smem_rowsum_1 : smem_rowsum_0;
    float *smem_scratchpad =
        (tile_k & 1) ? smem_scratchpad_1 : smem_scratchpad_0;

    const auto spad_addr_Q = spad_addr_Q0;
    const auto spad_addr_K_consume = (tile_k & 1) ? spad_addr_K1 : spad_addr_K0;
    const auto spad_addr_K_produce = (tile_k & 1) ? spad_addr_K0 : spad_addr_K1;
    const auto spad_addr_V_consume = (tile_k & 1) ? spad_addr_V1 : spad_addr_V0;
    const auto spad_addr_V_produce = (tile_k & 1) ? spad_addr_V0 : spad_addr_V1;
    const auto spad_addr_S_consume = (tile_k & 1) ? spad_addr_S1 : spad_addr_S0;
    const auto spad_addr_S_produce = (tile_k & 1) ? spad_addr_S0 : spad_addr_S1;
    const auto spad_addr_P_consume = (tile_k & 1) ? spad_addr_P1 : spad_addr_P0;
    const auto spad_addr_P_produce = (tile_k & 1) ? spad_addr_P0 : spad_addr_P1;
    const auto spad_addr_O = spad_addr_O0; // NOTE: there's only single O tile
    asm volatile ("dbuf_sel_end_%=:" :: );

    // GEMM II: O = O + P*V
    // --------------------
    // This is done *before* GEMM I in the software pipeline, working on the
    // online softmax result tile from the previous iteration

    if (tile_k >= 2) // delay by 2 iters for pipelining
    {
      const uint32_t tile_k_ = tile_k - 2;

      asm volatile("gemm_pv_start_%=:" ::);

      if (tid_in_warpgroup == 0) {
#if 0
      if (tile_k_ == 0) {
        gemmini_fence();
        GEMMINI_CISC_CMD_I(0);
      } else if (tile_k_ & 1) {
        gemmini_fence();
        GEMMINI_CISC_CMD_I(2);
      } else {
        gemmini_fence();
        GEMMINI_CISC_CMD_I(1);
      }
#else
        // do matmul
        // among other things, this also configures CONFIG_BOUNDS so that the
        // DMA knows the full matrix dimensions
        sp_tiled_matmul_full_spad_ws(
            spad_addr_P_consume, spad_addr_V_consume,
            /*spad_D=*/0, /*spad_C=*/spad_addr_O,
            /*I=*/(B_ROW / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
            /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_matmul);
#endif

        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();

        // mvout to SMEM
        // GEMMINI_CISC_CMD_I(9);
        sp_tiled_matmul_full_spad_ws(
            /*spad_A=*/spad_addr_P_consume, /*spad_B=*/spad_addr_V_consume,
            /*spad_D=*/0, /*spad_C=*/spad_addr_O,
            /*I=*/(B_ROW / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
            /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_mvout_spad);
        gemmini_fence();

        if constexpr (DEBUG) {
          // for copy-out to GMEM
          gemmini_fence();
        }
      }

      // reconverge from mmio divergence
      threadblock_barrier(warpgroup_id_in_cluster,
                          warps_per_warpgroup_per_core);

      asm volatile("gemm_pv_finish_%=:" ::);

      if constexpr (DEBUG) {
        if (warpgroup_id == 0) {
          // O after PV
          if (tile_k_ == 0) {
            thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                smem_O, gmem_tmp_d6, tid_in_warpgroup, threads_per_warpgroup,
                warpgroup_id_in_cluster);
          } else if (tile_k_ == 1) {
            thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                smem_O, gmem_tmp_d7, tid_in_warpgroup, threads_per_warpgroup,
                warpgroup_id_in_cluster);
          }

          threadblock_barrier(warpgroup_id_in_cluster,
                              warps_per_warpgroup_per_core);
        }
      }
    }

    // GEMM I: S = Q*K
    //
    asm volatile("gemm_qk_start_%=:" ::);

    if (tid_in_warpgroup == 0) {
      // 0,2,.: opcode 0 (quartile 0/2, no accum)
      // 1,3,.: opcode 3 (quartile 1/3, no accum)
      const uint32_t opcode = 3 * (tile_k & 1);
      gemmini_fence();
      GEMMINI_CISC_CMD_I(opcode);

      gemmini_fence();
      gemmini_fence();
      gemmini_fence();
      gemmini_fence();

#if 0 // TODO: speed up mvout to SMEM
// loop_ws variant that skips configuring strides
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_resadd) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }
#endif

      // mvout to SMEM
      // GEMMINI_CISC_CMD_I(9);
      sp_tiled_matmul_full_spad_ws(
          /*spad_A=*/spad_addr_Q, /*spad_B=*/spad_addr_K_consume,
          /*spad_D=*/0, /*spad_C=*/spad_addr_S_produce,
          /*I=*/(B_ROW / DIM), /*J=*/(B_COL / DIM), /*K=*/(HEADDIM / DIM),
          /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
          /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
          /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_mvout_spad);
      gemmini_fence();

      if constexpr (DEBUG) {
        // for copy-out to GMEM
        gemmini_fence();
      }
    }

    // reconverge from mmio divergence
    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    asm volatile("gemm_qk_finish_%=:" ::);

    if constexpr (DEBUG) {
      if (warpgroup_id == 0) {
        if (tile_k == 0) {
          thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
              smem_S_produce, gmem_tmp_d0, tid_in_warpgroup, threads_per_warpgroup,
              warpgroup_id_in_cluster);
        } else if (tile_k == 1) {
          thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
              smem_S_produce, gmem_tmp_d1, tid_in_warpgroup, threads_per_warpgroup,
              warpgroup_id_in_cluster);
        }

        threadblock_barrier(warpgroup_id_in_cluster,
                            warps_per_warpgroup_per_core);
      }
    }

    if (tile_k >= 1) // delay by 1 iters for pipelining
    {
      const uint32_t tile_k_ = tile_k - 1;

      // Online softmax
      //
      thread_block_online_softmax</*block_row_major=*/GEMMINI_DMA>(
          smem_S_consume, smem_P_produce, tid_in_warpgroup,
          threads_per_warpgroup, warpgroup_id_in_cluster, smem_scratchpad,
          smem_rowmax, smem_rowsum, smem_O_row_scale);

      threadblock_barrier(warpgroup_id_in_cluster,
                          warps_per_warpgroup_per_core);

      if constexpr (DEBUG) {
        if (warpgroup_id == 0) {
          if (tile_k_ == 0) {
            thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e0, tid_in_warpgroup,
                                     threads_per_warpgroup,
                                     warpgroup_id_in_cluster);
            thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e2, tid_in_warpgroup,
                                     threads_per_warpgroup,
                                     warpgroup_id_in_cluster);
          } else if (tile_k_ == 1) {
            thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e1, tid_in_warpgroup,
                                     threads_per_warpgroup,
                                     warpgroup_id_in_cluster);
            thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e3, tid_in_warpgroup,
                                     threads_per_warpgroup,
                                     warpgroup_id_in_cluster);
          }

          threadblock_barrier(warpgroup_id_in_cluster,
                              warps_per_warpgroup_per_core);
        }
      }

      // TODO: put a synchronization here with GEMM-II

      // Oi rescale
      thread_block_O_rescale</*block_row_major=*/GEMMINI_DMA>(
          smem_O, smem_O /*in-place*/, smem_O_row_scale, tid_in_warpgroup,
          threads_per_warpgroup, warpgroup_id_in_cluster);

      // rescale-to-PV-GEMM barrier
      threadblock_barrier(warpgroup_id_in_cluster,
                          warps_per_warpgroup_per_core);

      if constexpr (DEBUG) {
        if (warpgroup_id == 0) {
          // O before PV
          if (tile_k_ == 0) {
            thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
                smem_P_produce, gmem_tmp_d2, tid_in_warpgroup,
                threads_per_warpgroup, warpgroup_id_in_cluster);
            thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                smem_O, gmem_tmp_d4, tid_in_warpgroup, threads_per_warpgroup,
                warpgroup_id_in_cluster);
          } else if (tile_k_ == 1) {
            thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
                smem_P_produce, gmem_tmp_d3, tid_in_warpgroup,
                threads_per_warpgroup, warpgroup_id_in_cluster);
            thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                smem_O, gmem_tmp_d5, tid_in_warpgroup, threads_per_warpgroup,
                warpgroup_id_in_cluster);
          }

          threadblock_barrier(warpgroup_id_in_cluster,
                              warps_per_warpgroup_per_core);
        }
      }
    }

    // data movement for K and V
    //
    // Q stays in SMEM for the entire loop
    asm volatile("move_k_v_start_%=:" ::);

    // NOTE: Beware of race conditions; with warp specialization, we need to
    // make sure below command code to DMA is not executed simultaneously
    // from the two warpgroups (which will result in hardware fault).
    // Currently the ping-pong scheduling scheme prevents that.
    if (tid_in_warpgroup == 0) {
      // configure GMEM addresses for K and V tiles
      // load K for the next iteration
      const float *gmem_K_tile = gmem_K + (B_COL * (tile_k + 1 /*runahead*/));
      // load V for the *previous* iteration; this will be consumed 2
      // iterations later
      const float *gmem_V_tile =
          gmem_V + (HEADDIM * B_COL * (tile_k - 1 /*dragbehind*/));
      ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_K_tile),
                               (uint64_t)(gmem_V_tile),
                               k_LOOP_WS_CONFIG_ADDRS_AB)
      // configure address strides for the DMA
      // FIXME: unnecessary?
      GEMMINI_CISC_CMD_R((HEADDIM /*V*/ << 20) | (dim_seqlen /*KT*/ << 8) |
                         8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
      gemmini_fence();

      // do DMA
      if (tile_k == 0) {
        // we load (k-1)th tile for V; skip V for the 1st iteration,
        sp_tiled_matmul_full_spad_ws(
            spad_addr_K_produce, spad_addr_V_produce,
            /*spad_D=*/0, /*spad_C=*/spad_addr_S_produce /*FIXME:bogus*/,
            /*I=*/(HEADDIM / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
            /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_only_a);
      } else {
        sp_tiled_matmul_full_spad_ws(
            spad_addr_K_produce, spad_addr_V_produce,
            /*spad_D=*/0, /*spad_C=*/spad_addr_S_produce /*FIXME:bogus*/,
            /*I=*/(HEADDIM / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
            /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
      }
      gemmini_fence();
    }

    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    asm volatile("move_k_v_finish_%=:" ::);
#if 0
#endif
  }

  asm volatile ("tile_loop_finish_%=:" :: );

  // // wait for warpgroup 1 to finish, which called the global barrier before
  // // entering the loop
  // if (warpgroup_id == 0) {
  //   threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);
  // }
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
