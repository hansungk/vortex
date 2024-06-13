#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "util.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define GEMMINI_DMA 0
#if SMEM_SIZE == 0x4000
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff001000)
#define SMEM_ADDR_Q2 ((float * const) 0xff002000)
#define SMEM_ADDR_Q3 ((float * const) 0xff003000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x80
#define SPAD_ADDR_Q2 0x100
#define SPAD_ADDR_Q3 0x180
#define BOUND_INST 0x400040004ULL
#elif SMEM_SIZE == 0x10000
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff004000)
#define SMEM_ADDR_Q2 ((float * const) 0xff008000)
#define SMEM_ADDR_Q3 ((float * const) 0xff00c000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x200
#define SPAD_ADDR_Q2 0x400
#define SPAD_ADDR_Q3 0x600
#define BOUND_INST 0x800080008ULL
#else
#error Unsupported smem size
#endif

// FIXME: NUM_THREADS and NUM_WARPS hardcoded
#if ((BM * BN / ELEM_PER_THREAD) > (CORES_PER_CLUSTER * 8 * 8))
#error "threadblock size too big for cluster"
#endif

inline void global_dmem_load(const uint32_t dim_n, const uint32_t dim_k,
                             const uint32_t k, const float *A, const float *B,
                             volatile float *local_a, volatile float *local_b,
                             const uint32_t tid_in_threadblock,
                             const uint32_t threadblock_id_x,
                             const uint32_t threadblock_id_y) {
  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_as_row = tid_in_threadblock / BM;
  const uint32_t local_as_col = tid_in_threadblock % BM;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;

  constexpr uint32_t threads_in_threadblock = (BM * BN) / ELEM_PER_THREAD;

  // Data move from GMEM to SMEM
  //
  // Make sure global offset values for A and B are contiguous between
  // neighboring threads to ensure GMEM coalescing.
  //
  // TODO: Sharedmem swizzling is important here
  if constexpr (!TRANSPOSE_AT_PRODUCE) {
    // FIXME: !TRANSPOSE_AS code is old

    const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
    // number of rows a full TB can read at a time
    constexpr uint32_t row_stride_a = threads_in_threadblock / BK;
    const float *global_a = A + dim_k * global_a_row + (k + local_a_col);
    volatile float *local_a_tmp = local_a + BK * local_a_row + local_a_col;

#pragma GCC unroll 1
    for (uint32_t local_row_offset = 0; local_row_offset < BM;
         local_row_offset += row_stride_a) {
      // const uint32_t global_a_offset =
      //     dim_k * (global_a_row + local_row_offset) + (k + local_a_col);
      // local_a[BK * (local_a_row + local_row_offset) + local_a_col] =
      //     A[global_a_offset];
      *local_a_tmp = *global_a;

      global_a += dim_k * row_stride_a;
      local_a_tmp += BK * row_stride_a;
    }
  } else {
    if constexpr (!GMEM_COALESCED_A) {
      constexpr uint32_t row_stride_as = threads_in_threadblock / BM;
      const uint32_t global_a_row = BM * threadblock_id_y + local_as_col;
      const float *global_a = A + dim_k * global_a_row + (k + local_as_row);
      // FIXME experimenting with global coalescing
      // const uint32_t global_a_row = BM * threadblock_id_y + local_as_row;
      // const float *global_a = A + dim_k * global_a_row + (k + local_as_col);
      volatile float *local_a_tmp = local_a + BM * local_as_row + local_as_col;

      static_assert(
          row_stride_as * 8 <= BK,
          "manual loop unrolling condition not met; consider increasing BK");
      static_assert(
          (BK % (row_stride_as * 8)) == 0,
          "manual loop unrolling condition not met; BK should be power-of-two");

#pragma GCC unroll 1
      for (uint32_t local_row_offset = 0; local_row_offset < BK;
           local_row_offset += row_stride_as * 8) {
        // @perf: bank conflicts here
        // const uint32_t global_a_offset =
        //     dim_k * (global_a_row) + (k + local_as_row + local_row_offset);
        // FIXME experimenting with global coalescing
        // const uint32_t global_a_offset =
        //     dim_k * (global_a_row + local_row_offset) + (k + local_as_col);
        // local_a[BM * (local_as_row + local_row_offset) + local_as_col] =
        //     A[global_a_offset];

        // *local_a_tmp = *global_a;
        asm volatile ("flw ft0, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft1, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft2, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft3, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft4, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft5, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft6, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft7, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;

        asm volatile ("fsw ft0, %0(%1)" :: "i"(BM * row_stride_as * 0 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft1, %0(%1)" :: "i"(BM * row_stride_as * 1 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft2, %0(%1)" :: "i"(BM * row_stride_as * 2 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft3, %0(%1)" :: "i"(BM * row_stride_as * 3 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft4, %0(%1)" :: "i"(BM * row_stride_as * 4 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft5, %0(%1)" :: "i"(BM * row_stride_as * 5 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft6, %0(%1)" :: "i"(BM * row_stride_as * 6 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft7, %0(%1)" :: "i"(BM * row_stride_as * 7 * sizeof(float)), "r"(local_a_tmp));
        local_a_tmp += BM * row_stride_as * 8;
      }
    } else {
      constexpr uint32_t row_stride_a = threads_in_threadblock / BK;
      const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
      const float *global_a = A + dim_k * global_a_row + (k + local_a_col);
      // NOTE that SMEM writes are transposed
      volatile float *local_a_tmp = local_a + BM * local_a_col + local_a_row;

      static_assert(
          row_stride_a * 8 <= BM,
          "manual loop unrolling condition not met; consider increasing BM");
      static_assert(
          (BM % (row_stride_a * 8)) == 0,
          "manual loop unrolling condition not met; BM should be power-of-two");

#pragma GCC unroll 1
      for (uint32_t local_row_offset = 0; local_row_offset < BM;
           local_row_offset += row_stride_a * 8) {
        // const uint32_t global_a_offset =
        //     dim_k * (global_a_row + local_row_offset) + (k + local_a_col);
        // NOTE that SMEM writes are transposed
        // local_a[BM * (local_a_col) + local_a_row + local_row_offset] =
        //     A[global_a_offset];

        asm volatile ("flw ft0, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft1, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft2, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft3, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft4, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft5, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft6, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft7, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;

        // stride along columns
        asm volatile ("fsw ft0, %0(%1)" :: "i"(row_stride_a * 0 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft1, %0(%1)" :: "i"(row_stride_a * 1 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft2, %0(%1)" :: "i"(row_stride_a * 2 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft3, %0(%1)" :: "i"(row_stride_a * 3 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft4, %0(%1)" :: "i"(row_stride_a * 4 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft5, %0(%1)" :: "i"(row_stride_a * 5 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft6, %0(%1)" :: "i"(row_stride_a * 6 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft7, %0(%1)" :: "i"(row_stride_a * 7 * sizeof(float)), "r"(local_a_tmp));
        local_a_tmp += row_stride_a * 8;
      }
    }
  }

  constexpr uint32_t row_stride_b = threads_in_threadblock / BN;
  const uint32_t global_b_col = BN * threadblock_id_x + local_b_col;
  const float *global_b = B + dim_n * (k + local_b_row) + global_b_col;
  volatile float *local_b_tmp = local_b + BN * local_b_row + local_b_col;

  static_assert(
      row_stride_b * 8 <= BK,
      "manual loop unrolling condition not met; consider increasing BK");
  static_assert(
      (BK % (row_stride_b * 8)) == 0,
      "manual loop unrolling condition not met; BK should be power-of-two");

#pragma GCC unroll 1
  for (uint32_t load_offset = 0; load_offset < BK;
       load_offset += row_stride_b * 8) {
    // const uint32_t global_b_offset =
    //     dim_n * (k + local_b_row + load_offset) + global_b_col;
    // local_b[BN * (local_b_row + load_offset) + local_b_col] =
    //     B[global_b_offset];

    // *local_b_tmp = *global_b;

    // global_b += dim_n * row_stride_b;
    // local_b_tmp += BN * row_stride_b;

    asm volatile ("flw ft0, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft1, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft2, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft3, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft4, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft5, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft6, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft7, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;

    asm volatile ("fsw ft0, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft1, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
    asm volatile ("fsw ft2, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft3, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
    asm volatile ("fsw ft4, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft5, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
    asm volatile ("fsw ft6, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft7, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
  }
}

inline void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threads_per_threadblock,
                              const uint32_t threadblock_dim_y,
                              /*const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,*/
                              const uint32_t threadblocks_per_cluster,
                              const uint32_t threadblock_id_in_cluster,
                              float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_as_row = tid_in_threadblock / BM;
  const uint32_t local_as_col = tid_in_threadblock % BM;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;

  // no double-buffering
  const uint32_t threads_per_warpgroup = threads_per_threadblock;
  const uint32_t warp_id_in_warpgroup = tid_in_threadblock / NUM_THREADS;
  const uint32_t warp_row = warp_id_in_warpgroup / (BN / WN);
  const uint32_t warp_col = warp_id_in_warpgroup % (BN / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;

  volatile float *local_a = sharedmem_per_threadblock;
  constexpr size_t local_a_elems = (BM * BK);
  volatile float *local_b = sharedmem_per_threadblock + local_a_elems;
  constexpr size_t local_b_elems = (BK * BN);

  volatile float *local_a_buf = local_b + local_b_elems;
  volatile float *local_b_buf = local_a_buf + local_a_elems;

  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);

#if (GEMMINI_DMA == 1)
  if (tid_in_threadblock == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose,
    // b_transpose);

    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 1);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);

    gemmini_fence();
  }
#endif

  // divide rows (M) by the number of threadblocks
  const uint32_t dim_m_range = (dim_m / threadblocks_per_cluster);
  const uint32_t dim_m_start = dim_m_range * threadblock_id_in_cluster;
  const uint32_t block_m_start = dim_m_start / BM;
  const uint32_t block_m_end = (dim_m_start + dim_m_range) / BM;

#pragma GCC unroll 1
  for (uint32_t block_m = block_m_start; block_m < block_m_end; block_m++) {
#pragma GCC unroll 1
    for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {
      // clear out C
      initialize_C(0);
      initialize_C(1);

      if constexpr (GEMMINI_DMA) {
        // pipeline initiation
        if (tid_in_threadblock == 0) {
          // configure dma gmem address to load from
          // FIXME: block_k is wrong
          ROCC_INSTRUCTION_RS1_RS2(
              XCUSTOM_ACC,
              (uint64_t)(A + block_m * BM * dim_k + /*block_k:*/0 * BK),
              (uint64_t)(B + /*block_k:*/0 * BK * dim_n + block_n * BN),
              k_LOOP_WS_CONFIG_ADDRS_AB)
          // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
          GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);
          gemmini_fence();

          GEMMINI_CISC_CMD_I(12);
          gemmini_fence();

#if 0
          // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
          // FIXME: block_k is 0 for two times
          sp_tiled_matmul_full_spad_ws(
#if 1
              SPAD_ADDR_Q0, SPAD_ADDR_Q1,
#else
              (/*block_k:*/ 0 & 1) ? SPAD_ADDR_Q2 : SPAD_ADDR_Q0,
              (/*block_k:*/ 0 & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q1,
#endif
              /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q3,
              /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
              /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
          gemmini_fence();
#endif
        }

        threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
      }

#pragma GCC unroll 1
      for (uint32_t block_k = 0; (block_k * BK) < (dim_k); block_k++) {

        // producer code: GMEM->SMEM memory movement
        // ---------------------------------------------------------------------
        //
        // this is either done using DMA or SIMT cores depending on GEMMINI_DMA

#if (GEMMINI_DMA == 1)
        if (tid_in_threadblock == 0) {
          // configure dma gmem address to load from
          // FIXME: block_k is wrong
          ROCC_INSTRUCTION_RS1_RS2(
              XCUSTOM_ACC,
              (uint64_t)(A + block_m * BM * dim_k + (block_k + 1/*runahead*/) * BK),
              (uint64_t)(B + (block_k + 1/*runahead*/) * BK * dim_n + block_n * BN),
              k_LOOP_WS_CONFIG_ADDRS_AB)
          // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
          GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);
          // gemmini_fence();

          // block_k is even: opcode 13 (write to local_a_buf)
          // block_k is odd:  opcode 12 (write to local_a)
          const uint32_t opcode = 13 - (block_k & 1);
          GEMMINI_CISC_CMD_R(opcode);
          // // TODO: branch is probably slow
          // if (block_k & 1) {
          //   GEMMINI_CISC_CMD_I(12);
          // } else { // block_k == 0 is here
          //   GEMMINI_CISC_CMD_I(13);
          // }

          // configure loop iteration bounds
          // FIXME: shouldn't be necessary
          // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST,
          // k_LOOP_WS_CONFIG_BOUNDS) ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
          // SPAD_ADDR_Q0, SPAD_ADDR_Q1, k_LOOP_WS_CONFIG_SPAD_AB)
          // ROCC_INSTRUCTION_RS1_RS2(
          //     XCUSTOM_ACC,
          //     ((uint64_t)(/*a_spad_id:*/ 0) << 18) |
          //         ((uint64_t)(/*b_spad_id:*/ 0) << 16) |
          //         ((uint64_t)(/*act:0*/ 0) << 8) | ((/*low_D:*/ 0) << 2) |
          //         ((/*full_C:*/ 0) << 1) | (/*ex_accumulate:*/ 0),
          //     ((uint64_t)(/*C_spad_addr:*/ A) << 32) | 0x200U | (skips) |
          //         ((/*is_resadd*/ 0) << 2) | ((/*B_transpose:*/ 0) << 1) |
          //         (/*A_transpose:*/ 1),
          //     k_LOOP_WS)
          // gemmini_fence();

#if 0
          uint32_t spad_a_produce;
          uint32_t spad_b_produce;
          const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
          const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
          spad_a_produce =
              ((mask_odd & (SPAD_ADDR_Q0)) | (mask_even & (SPAD_ADDR_Q2)));
          spad_b_produce =
              ((mask_odd & (SPAD_ADDR_Q1)) | (mask_even & (SPAD_ADDR_Q3)));
          // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
          // FIXME: block_k is 0 for two times
          sp_tiled_matmul_full_spad_ws(
              spad_a_produce,
              spad_b_produce,
              /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
              /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
              /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
#endif
        }
#else
        global_dmem_load(dim_n, dim_k, block_k * BK, A, B, local_a, local_b,
                         tid_in_threadblock, block_n, block_m);

        threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
#endif

        // consumer code: SMEM->RF and compute
        // ----------------------------------------------------------------------
        // @perf: this loop spills to stack a lot because of all the flws in
        const volatile float *local_a_consume;
        const volatile float *local_b_consume;
        if constexpr (GEMMINI_DMA) {
          // local_a_consume = (k_index % 2) ? local_a_buf : local_a;
          // local_b_consume = (k_index % 2) ? local_b_buf : local_b;
          // FIXME: swap multiply with bitshifts
          // const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
          // const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
          // local_a_consume = reinterpret_cast<volatile float *>(
          //     (mask_odd & reinterpret_cast<uintmax_t>(local_a_buf)) |
          //     (mask_even & reinterpret_cast<uintmax_t>(local_a)));
          // local_b_consume = reinterpret_cast<volatile float *>(
          //     (mask_odd & reinterpret_cast<uintmax_t>(local_b_buf)) |
          //     (mask_even & reinterpret_cast<uintmax_t>(local_b)));
          local_a_consume = local_a + (block_k & 1) * (local_a_elems + local_b_elems);
          local_b_consume = local_b + (block_k & 1) * (local_a_elems + local_b_elems);
        } else {
          local_a_consume = local_a;
          local_b_consume = local_b;
        }

#pragma GCC unroll 1
        for (int i = 0; i < BK_LOOP; i++) {
#pragma GCC unroll 4
          for (uint32_t local_k = 0; local_k < BK; local_k += TCK) {
#pragma GCC unroll 2
            for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
              // SMEM -> RF
              vx_wmma_load_b(local_b_consume, local_k, warp_col, wn_iter, tid_in_warp);
#pragma GCC unroll 2
              for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
                // SMEM -> RF
                vx_wmma_load_a(local_a_consume, local_k, warp_row, wm_iter,
                               tid_in_warp);
                // perform mma
                vx_wmma(wm_iter);
              }
            }
          }
        }

        if constexpr (GEMMINI_DMA) {
          // Call gemmini fence at the end of the loop to overlap dma & wmma.
          // Hopefully by this time, dma would have finished so that this is a
          // no-op
          if (tid_in_threadblock == 0) {
            gemmini_fence();
          }
        }

        threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
      }

#pragma GCC unroll 2
      for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll 2
        for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
          write_results(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter,
                        dim_n, C, block_n, block_m);
        }
      }
    }
  }
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

  const uint32_t threadblock_dim_y = vx_num_warps() / threadblocks_per_cluster;
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
  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR + (GEMMINI_DMA ? 2 /*double-buffer*/ : 1) *
                                         (2 * BM * BK) *
                                         threadblock_id_in_cluster;

  thread_block_gemm(arg, tid_in_threadblock, threads_per_threadblock,
                    threadblock_dim_y,
                    /*threadblock_id_x, threadblock_id_y,*/
                    threadblocks_per_cluster,
                    // threadblock_id,
                    threadblock_id_in_cluster,
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
