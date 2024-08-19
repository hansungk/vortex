#ifndef _SGEMM_IMPL_H_
#define _SGEMM_IMPL_H_

#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define FP_SIZE 32

// "fake" fp16 type that only has the correct data width.
using float16_t = uint16_t;

#if (FP_SIZE == 32)
using float_type = float;
#elif (FP_SIZE == 16)
using float_type = float16_t;
#endif

// Constraints on parameters:
// * Memory:
//   (BM + BN) * BK * sizeof(T) <= sharedmem size.
//   BM * BK == BN * BK >= threadblock size >= NT * CORES_PER_CLUSTER
//     When larger, the kernel runs a sequential loop to read into sharedmem;
//     but smaller case is not handled.
// * Compute:
//   ( M* N) / (TM*TN) == grid size >= NC*NW*NT
//   (BM*BN) / (TM*TN) == threadblock size < NT * NW * CORES_PER_CLUSTER
//   (BM*BN) / (TM*TN) == threadblock size >= NT * CORES_PER_CLUSTER
// * Combining BM * BK >= (BM*BN) / (TM*TN) == threadblock yields
//   BM <= BK*TM*TN
#define BM 64
#define BN 64
#if (FP_SIZE == 32)
#define BK 64
#elif (FP_SIZE == 16)
#define BK 128
#else
#error "unsupported FP_SIZE"
#endif
#define WM 16
#define WN 8
#define TCM 8
#define TCN 8
#if (FP_SIZE == 32)
#define TCK 8
#elif (FP_SIZE == 16)
#define TCK 16
#else
#error "unsupported FP_SIZE"
#endif
#define WMITER (WM / TCM)
#define WNITER (WN / TCN)
#define ELEM_PER_THREAD (WM * WN / NUM_THREADS)
// FIXME: NUM_THREADS and NUM_WARPS hardcoded
#if ((BM * BN / ELEM_PER_THREAD) > (CORES_PER_CLUSTER * 8 * 8))
#error "threadblock size too big for cluster"
#endif

// number of loop around the inner 0..TCK..BK loop to simulate perfect-DRAM
// scenario
#define BK_LOOP 1
// Whether to transpose smem A tile at GMEM->SMEM (produce), or SMEM->RF
// (consume).  This is because the tensor core expects the A tile to be stored
// in column-major order in SMEM, whereas it will be ultimately stored in
// row-major in the RF.
//
// For correctness, only one of either should be 1.  E.g., PRODUCE 1 CONSUME 0
// generates the NN kernel where both A and B are stored row-major in GMEM.
// To model the case where the A matrix is already stored column-major in GMEM,
// set both to 0.
#define TRANSPOSE_AT_PRODUCE 0
#define TRANSPOSE_AT_CONSUME 0

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

enum class MemLayout {
  MN_major,
  K_major,
};

inline constexpr void map_operand_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // Figure 7(a) in paper
  // row  0~ 3: threadgroups 0 and 2
  // row  4~ 7: threadgroups 4 and 6
  // row  8~11: threadgroups 1 and 3
  // row 12~15: threadgroups 5 and 7
  row = tid % 4;
  row += (tg * 8) % 16;
  row += (tg / 4) * 4;

  // B (column major)
  // NOTE: Matrix B mapping in Figure 7(a) is incorrect; below is the
  // corrected mapping:
  // col  0~ 3: threadgroups 0 and 1
  // col  4~ 7: threadgroups 4 and 5
  // col  8~11: threadgroups 2 and 3
  // col 12~15: threadgroups 6 and 7
  col = tid % 4;
  col += ((tg % 4) / 2) * 8;
  col += (tg / 4) * 4;
}

inline constexpr void map_operand_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // row  0~ 3: threadgroup 0
  // row  4~ 7: threadgroup 1
  row = tid % 4;
  row += tg * 4;

  // B (column major)
  // col  0~ 3: threadgroup 0
  // col  4~ 7: threadgroup 1
  col = tid % 4;
  col += tg * 4;
}

inline constexpr void map_operand(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_operand_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_operand_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

inline constexpr void map_c_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  // Figure 7(b), left
  col = ((tg % 4) / 2) * 8;
  row = (tg * 8) % 16;
  row += (tg / 4) * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  col = 0;
  row = tg * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_c_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_c_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

#define RISCV_CUSTOM3   0x7B

inline void vx_wmma(const int dest_reg) {
  if (dest_reg == 0) {
    asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
  } else {
    asm volatile (".insn r %0, 0, 0, x1, x0, x0" :: "i"(RISCV_CUSTOM3));
  }
}

// `local_k` is assumed to be multiple of TCK
template <typename T, MemLayout layout>
inline void wmma_load_a(volatile const T *smem_A, const int local_k,
                        const int warp_row, const int wm_iter,
                        const int thread_in_warp) {
  asm volatile ("wmma_load_a_start_%=:" :: );

  const int tid = thread_in_warp;
  const int tg = tid / 4;

  // @perf: this is duplicately computed in wmma_load_a and wmma_load_b
  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  // In fp16 mode, bit-pack two fp16 elements into each fp32 element, and do
  // data movement at the fp32 granularity.  Assuming that the matrix is stored
  // row-major in GMEM, the packed fp16 pairs belong to the same row,
  // neighboring columns; therefore, it essentially becomes equivalent to
  // moving a fp32 matrix whose column dimensions (dim_k/BK/k) are compressed
  // by a factor of two.
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  constexpr int BK_adjusted = BK / packed_factor;
  const int local_k_adjusted = local_k / packed_factor;

  if constexpr (layout == MemLayout::K_major) {
    constexpr int smem_A_rows = BM;
    constexpr int smem_A_cols = BK_adjusted;

    // int A_offset = (WM * warp_row + TCM * wm_iter + row) * smem_A_cols;

    // f8-f15 stores a single row of A
    const volatile uint8_t *smem_addr;
    smem_addr = reinterpret_cast<const volatile uint8_t *>(
        &reinterpret_cast<const volatile float *>(
            smem_A)[(WM * warp_row + TCM * wm_iter + row) * smem_A_cols +
                    local_k /* FIXME: adjust for fp16? */]);
    // step to the next column
    // @perf: bank conflicts; threads read from different rows
    asm volatile("flw  f0, %0(%1)" ::"i"(0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" ::"i"(1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" ::"i"(2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" ::"i"(3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" ::"i"(4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" ::"i"(5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" ::"i"(6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" ::"i"(7 * sizeof(float)), "r"(smem_addr));
  } else if (layout == MemLayout::MN_major) {
    constexpr int smem_AS_rows = BK_adjusted;
    constexpr int smem_AS_cols = BM;

    const volatile uint8_t *smem_addr;
    smem_addr = reinterpret_cast<const volatile uint8_t *>(
        &reinterpret_cast<const volatile float *>(
            smem_A)[((local_k_adjusted + 0) * smem_AS_cols) +
                    (WM * warp_row + TCM * wm_iter) + row]);
    // f8-f15 stores a single row of A
    // threads read from different columns; no bank conflicts
    asm volatile("flw  f0, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" :: "i"(smem_AS_cols * 4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" :: "i"(smem_AS_cols * 5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" :: "i"(smem_AS_cols * 6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" :: "i"(smem_AS_cols * 7 * sizeof(float)), "r"(smem_addr));
  } else {
    static_assert(layout ==
                      MemLayout::K_major /* fake cond that is always false */,
                  "unsupported memory layout");
  }

  asm volatile ("wmma_load_a_finish_%=:" :: );
}

// `local_k` is assumed to be multiple of TCK
template <typename T, MemLayout layout>
inline void wmma_load_b(const volatile T *smem_B, const int local_k,
                           const int warp_col, const int wn_iter,
                           const int thread_in_warp) {
  asm volatile ("wmma_load_b_start_%=:" :: );

  static_assert(layout == MemLayout::MN_major,
                "only N-major layout for the B tile is supported");

  const int tid = thread_in_warp;
  const int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  // see comment in wmma_load_a
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  constexpr int BK_adjusted = BN / packed_factor;
  constexpr int BN_adjusted = BN / packed_factor;
  const int local_k_adjusted = local_k / packed_factor;

  // B is stored N-major in smem
  constexpr int smem_B_rows = BK_adjusted;
  constexpr int smem_B_cols = BN;

  const volatile uint8_t *smem_addr;
  smem_addr = reinterpret_cast<const volatile uint8_t *>(
      &reinterpret_cast<const volatile float *>(
          smem_B)[((local_k_adjusted + 0) * smem_B_cols) +
                  (WN * warp_col + TCN * wn_iter) + col]);
  // f8-f15 stores a single column of B
  // threads read from different columns; no bank conflicts
  asm volatile("flw  f8, %0(%1)" :: "i"(smem_B_cols * 0 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw  f9, %0(%1)" :: "i"(smem_B_cols * 1 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f10, %0(%1)" :: "i"(smem_B_cols * 2 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f11, %0(%1)" :: "i"(smem_B_cols * 3 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f12, %0(%1)" :: "i"(smem_B_cols * 4 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f13, %0(%1)" :: "i"(smem_B_cols * 5 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f14, %0(%1)" :: "i"(smem_B_cols * 6 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f15, %0(%1)" :: "i"(smem_B_cols * 7 * sizeof(float)), "r"(smem_addr));

  asm volatile ("wmma_load_b_finish_%=:" :: );
}

inline void initialize_C(const int dest_reg) {
  // initialize C to zeros
  if (dest_reg == 0) {
    asm volatile("fmv.w.x f16, x0");
    asm volatile("fmv.w.x f17, x0");
    asm volatile("fmv.w.x f18, x0");
    asm volatile("fmv.w.x f19, x0");
    asm volatile("fmv.w.x f20, x0");
    asm volatile("fmv.w.x f21, x0");
    asm volatile("fmv.w.x f22, x0");
    asm volatile("fmv.w.x f23, x0");
  } else {
    asm volatile("fmv.w.x f24, x0");
    asm volatile("fmv.w.x f25, x0");
    asm volatile("fmv.w.x f26, x0");
    asm volatile("fmv.w.x f27, x0");
    asm volatile("fmv.w.x f28, x0");
    asm volatile("fmv.w.x f29, x0");
    asm volatile("fmv.w.x f30, x0");
    asm volatile("fmv.w.x f31, x0");
  }
}

__attribute__((always_inline)) inline void
wmma_store(const int thread_in_warp, const int warp_col, const int warp_row,
           const int wn_iter, const int wm_iter, const int dim_n,
           float *write_addr) {
  asm volatile ("wmma_store_start_%=:" :: );

  int tid = thread_in_warp;

  // these are [0, TCM/TCN)
  int tid_row = 0;
  int tid_col = 0;
  map_c(tid, tid_row, tid_col);

  int local_row = (WM * warp_row + TCM * wm_iter) + tid_row;
  int local_col = (WN * warp_col + TCN * wn_iter) + tid_col;

  // @perf: this likely causes a lot of gmem bank conflicts
  if (wm_iter == 0) {
    volatile uint8_t *addr = reinterpret_cast<volatile uint8_t *>(
        &write_addr[dim_n * (local_row + 0) + (local_col + 0)]);
    volatile uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
    asm volatile("fsw f16, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr));
    asm volatile("fsw f17, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr));
    asm volatile("fsw f18, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr_tworow));
    asm volatile("fsw f19, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr_tworow));
    asm volatile("fsw f20, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr));
    asm volatile("fsw f21, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr));
    asm volatile("fsw f22, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr_tworow));
    asm volatile("fsw f23, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr_tworow));
  } else {
    volatile uint8_t *addr = reinterpret_cast<volatile uint8_t *>(
        &write_addr[dim_n * (local_row + 0) + (local_col + 0)]);
    volatile uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
    asm volatile("fsw f24, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr));
    asm volatile("fsw f25, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr));
    asm volatile("fsw f26, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr_tworow));
    asm volatile("fsw f27, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr_tworow));
    asm volatile("fsw f28, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr));
    asm volatile("fsw f29, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr));
    asm volatile("fsw f30, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr_tworow));
    asm volatile("fsw f31, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr_tworow));
  }

  asm volatile ("wmma_store_finish_%=:" :: );
}

inline void threadblock_barrier(const uint32_t barrier_id, const uint32_t count) {
  vx_fence();
  vx_barrier(barrier_id, count);
}

// Move a single matrix tile from global memory (GMEM) to shared memory (SMEM).
// `dim_col`: column dimension of the global matrix.
template <typename T,
          MemLayout gmem_layout, // memory layout of the GMEM tile
          MemLayout smem_layout, // memory layout of the GMEM tile
          uint32_t tile_dim_mn,  // row dimension of the SMEM tile
          uint32_t tile_dim_k    // column dimension of the SMEM tile
          >
__attribute__((always_inline)) inline void
load_tile_to_smem(const uint32_t dim_col, const uint32_t mn_index,
                  const uint32_t k, const T *global_addr,
                  volatile T *local_addr, const uint32_t tid_in_threadblock) {
  asm volatile("global_dmem_load_start_new_%=:" ::);

  // In fp16 mode, bit-pack two fp16 elements into each fp32 element, and do
  // data movement at the fp32 granularity.  The tensor core hardware assumes
  // the fp16 elements are contiguously stored along the K-dimension;
  // therefore, this essentially becomes equivalent to a fp32 GEMM where the
  // K-dimension is shrinked by the factor of two.
  constexpr uint32_t packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);

  constexpr uint32_t tile_dim_k_packed = tile_dim_k / packed_factor;
  constexpr uint32_t gmem_dim_row =
      (gmem_layout == MemLayout::K_major) ? tile_dim_mn : tile_dim_k_packed;
  constexpr uint32_t gmem_dim_col =
      (gmem_layout == MemLayout::K_major) ? tile_dim_k_packed : tile_dim_mn;
  constexpr uint32_t smem_dim_row =
      (smem_layout == MemLayout::K_major) ? tile_dim_mn : tile_dim_k_packed;
  constexpr uint32_t smem_dim_col =
      (smem_layout == MemLayout::K_major) ? tile_dim_k_packed : tile_dim_mn;

  const uint32_t dim_col_ =
      (gmem_layout == MemLayout::K_major) ? dim_col / packed_factor : dim_col;
  // FIXME: unsure about this
  const uint32_t k_ = k / packed_factor;

  // threads in the threadblock always do contiguous accesses in the gmem
  const uint32_t local_row_gmem = tid_in_threadblock / gmem_dim_col;
  const uint32_t local_col_gmem = tid_in_threadblock % gmem_dim_col;

  constexpr bool transposed_write = (gmem_layout != smem_layout);
  // if transposed, threads write to smem in reversed col/row
  const uint32_t local_row_smem =
      transposed_write ? local_col_gmem : local_row_gmem;
  const uint32_t local_col_smem =
      transposed_write ? local_row_gmem : local_col_gmem;

  // FIXME: don't hardcode this here
  constexpr uint32_t threads_per_threadblock = (BM * BN) / ELEM_PER_THREAD;

  const uint32_t global_row_mn_major = k_ + local_row_gmem;
  const uint32_t global_col_mn_major = smem_dim_col * mn_index + local_col_gmem;
  const uint32_t global_row_k_major = gmem_dim_row * mn_index + local_row_gmem;
  const uint32_t global_col_k_major = k_ + local_col_gmem;
  const uint32_t global_row = (gmem_layout == MemLayout::K_major)
                                  ? global_row_k_major
                                  : global_row_mn_major;
  const uint32_t global_col = (gmem_layout == MemLayout::K_major)
                                  ? global_col_k_major
                                  : global_col_mn_major;

  const float *global = reinterpret_cast<const float *>(global_addr) +
                        dim_col_ * global_row + global_col;
  volatile float *local = reinterpret_cast<volatile float *>(local_addr) +
                          smem_dim_col * local_row_smem + local_col_smem;

  constexpr uint32_t row_stride = threads_per_threadblock / gmem_dim_col;
  static_assert(row_stride * 8 <= gmem_dim_row,
                "manual loop unrolling condition not met; tile row dimension "
                "is too shallow");
  static_assert((gmem_dim_row % (row_stride * 8)) == 0,
                "manual loop unrolling condition not met; tile row dimension "
                "should be power-of-two");

#pragma GCC unroll 1
  // loop-unrolled flw/fsw to increase reuse distance and IPC
  for (uint32_t load_offset = 0; load_offset < gmem_dim_row;
       load_offset += row_stride * 8) {
    // equivalent code:
    //
    // *local = *global;
    // global += dim_col * row_stride;
    // local += BN * row_stride;

    // read same-column elements into fp registers
    asm volatile("flw ft0, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft1, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft2, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft3, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft4, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft5, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft6, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;
    asm volatile("flw ft7, (%0)" ::"r"(global));
    global += dim_col_ * row_stride;

    // need to branch because address offset constant in the inline assembly
    // cannot be larger than a certain limit
    if constexpr (!transposed_write) {
      asm volatile("fsw ft0, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft1, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride * 2;
      asm volatile("fsw ft2, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft3, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride * 2;
      asm volatile("fsw ft4, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft5, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride * 2;
      asm volatile("fsw ft6, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft7, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride * 2;
    } else {
      // currently, tensor core hardware only supports MN-major SMEM tile
      // layout for correct results
      static_assert(gmem_layout == MemLayout::K_major);
      static_assert(smem_layout == MemLayout::MN_major);

      asm volatile("fsw ft0, %0(%1)" ::"i"(row_stride * 0 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft1, %0(%1)" ::"i"(row_stride * 1 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft2, %0(%1)" ::"i"(row_stride * 2 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft3, %0(%1)" ::"i"(row_stride * 3 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft4, %0(%1)" ::"i"(row_stride * 4 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft5, %0(%1)" ::"i"(row_stride * 5 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft6, %0(%1)" ::"i"(row_stride * 6 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft7, %0(%1)" ::"i"(row_stride * 7 * sizeof(float)),
                   "r"(local));
      local += row_stride * 8;
    }
  }

  asm volatile("global_dmem_load_finish_new_%=:" ::);
}

// Do a single tile*tile matrix multiplication using the matrix data stored in
// SMEM.  Useful in fused kernels where GEMMs are done at a per-tile scope.
template <typename T,
          MemLayout layout_a,        // memory layout of `local_a`
          MemLayout layout_b,        // memory layout of `local_b`
          bool write_to_smem = false // if true, write result tile to SMEM at a
                                     // given address
          >
__attribute__((always_inline)) inline void
thread_block_gemm_single_tile(const T *local_a, const T *local_b, T *local_c,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threads_per_threadblock) {
  // no double-buffering
  // FIXME: duplicated from thread_block_gemm
  const uint32_t threads_per_warpgroup = threads_per_threadblock;
  const uint32_t warp_id_in_warpgroup = tid_in_threadblock / NUM_THREADS;
  const uint32_t warp_row = warp_id_in_warpgroup / (BN / WN);
  const uint32_t warp_col = warp_id_in_warpgroup % (BN / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;

#pragma GCC unroll 1
  for (int i = 0; i < BK_LOOP; i++) {
#pragma GCC unroll 4
    for (uint32_t local_k = 0; local_k < BK; local_k += TCK) {
#pragma GCC unroll 2
      for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
        // SMEM -> RF
        wmma_load_b<T, layout_b>(local_b, local_k, warp_col, wn_iter,
                                 tid_in_warp);
#pragma GCC unroll 2
        for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
          // SMEM -> RF
          wmma_load_a<T, layout_a>(local_a, local_k, warp_row, wm_iter,
                                   tid_in_warp);
          // perform mma
          vx_wmma(wm_iter);
        }
      }
    }
  }

  if constexpr (GEMMINI_DMA) {
    // Call gemmini fence at the end of the loop to overlap dma & wmma.
    // Usually, by this time, dma has finished the copy so that this
    // becomes a no-op.
    if (tid_in_threadblock == 0) {
      gemmini_fence();
    }
  }

  if constexpr (write_to_smem) {
#pragma GCC unroll
    for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll
      for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
        wmma_store(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter, BN,
                   local_c);
      }
    }
  }
}

template <typename T, bool write_to_gmem = true,
          // by default, A/B tiles are placed at the start of the smem
          uint32_t smem_a_offset = 0,      // byte offset of A tile in shared
                                           // memory
          uint32_t smem_a_dbuf_offset = 0, // byte offset of A
                                           // double-buffer tile in shared
                                           // memory
          uint32_t smem_b_offset = sizeof(float) * BM *
                                   BK, // byte offset of B tile
                                       // in shared memory
          uint32_t smem_b_dbuf_offset = sizeof(float) * BM *
                                        BK // byte offset of B double-buffer
                                           // tile in shared memory
          >
inline void thread_block_gemm(const T *A, const T *B, float *C,
                              const uint32_t dim_m, const uint32_t dim_n,
                              const uint32_t dim_k,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threads_per_threadblock,
                              const uint32_t threadblocks_per_cluster,
                              const uint32_t threadblock_id_in_cluster,
                              uint8_t *sharedmem_per_threadblock) {
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
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threads_per_threadblock;

  volatile T *local_a =
      reinterpret_cast<T *>(sharedmem_per_threadblock + smem_a_offset);
  volatile T *local_a_buf =
      reinterpret_cast<T *>(sharedmem_per_threadblock + smem_a_dbuf_offset);
  volatile T *local_b =
      reinterpret_cast<T *>(sharedmem_per_threadblock + smem_b_offset);
  volatile T *local_b_buf =
      reinterpret_cast<T *>(sharedmem_per_threadblock + smem_b_dbuf_offset);

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

          GEMMINI_CISC_CMD_I(10);
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

        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
      }

#pragma GCC unroll 1
      for (uint32_t block_k = 0; (block_k * BK) < dim_k; block_k++) {

        // producer code: GMEM->SMEM memory movement
        // ---------------------------------------------------------------------
        //
        // this is either done using DMA or SIMT cores depending on GEMMINI_DMA

#if (GEMMINI_DMA == 1)
        if ((tid_in_threadblock == 0) && ((block_k * BK) != (dim_k - BK))) {
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

          // block_k is even: opcode 11 (write to local_a_buf)
          // block_k is odd:  opcode 10 (write to local_a)
          const uint32_t opcode = 11 - (block_k & 1);
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
        // move A
        if constexpr (!TRANSPOSE_AT_PRODUCE) {
          load_tile_to_smem<T, MemLayout::MN_major, MemLayout::MN_major, BM,
                            BK>(dim_m, block_m, block_k * BK, A, local_a,
                                tid_in_threadblock);
        } else {
          load_tile_to_smem<T, MemLayout::K_major, MemLayout::MN_major, BM, BK>(
              dim_k, block_m, block_k * BK, A, local_a, tid_in_threadblock);
        }

        // move B
        load_tile_to_smem<T, MemLayout::MN_major, MemLayout::MN_major, BN, BK>(
            dim_n, block_n, block_k * BK, B, local_b, tid_in_threadblock);

        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
#endif

        // consumer code: SMEM->RF and compute
        // ----------------------------------------------------------------------
        // @perf: this loop spills to stack a lot because of all the flws in
        const volatile T *local_a_consume;
        const volatile T *local_b_consume;
        if constexpr (GEMMINI_DMA) {
          // local_a_consume = (k_index % 2) ? local_a_buf : local_a;
          // local_b_consume = (k_index % 2) ? local_b_buf : local_b;
          // FIXME: swap multiply with bitshifts
          // const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
          // const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
          // local_a_consume = reinterpret_cast<volatile T *>(
          //     (mask_odd & reinterpret_cast<uintmax_t>(local_a_buf)) |
          //     (mask_even & reinterpret_cast<uintmax_t>(local_a)));
          // local_b_consume = reinterpret_cast<volatile T *>(
          //     (mask_odd & reinterpret_cast<uintmax_t>(local_b_buf)) |
          //     (mask_even & reinterpret_cast<uintmax_t>(local_b)));
          local_a_consume = local_a + (block_k & 1) * (BM * BK);
          local_b_consume = local_b + (block_k & 1) * (BK * BN);
        } else {
          // no double-buffering without DMA
          local_a_consume = local_a;
          local_b_consume = local_b;
        }

        constexpr MemLayout layout_a =
            TRANSPOSE_AT_CONSUME ? MemLayout::K_major : MemLayout::MN_major;
        thread_block_gemm_single_tile<T, layout_a, MemLayout::MN_major,
                                      /*write_to_smem=*/false>(
            local_a_consume, local_b_consume,
            static_cast<volatile T *>(nullptr) /*ignore*/, tid_in_threadblock,
            threads_per_threadblock);

        if constexpr (GEMMINI_DMA) {
          // Call gemmini fence at the end of the loop to overlap dma & wmma.
          // Usually, by this time, dma has finished the copy so that this
          // becomes a no-op.
          if (tid_in_threadblock == 0) {
            gemmini_fence();
          }
        }

        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
      }

      if constexpr (write_to_gmem) {
#pragma GCC unroll
        for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll
          for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
            float *global_offset_C = C + (BM * block_m) * dim_n + BN * block_n;
            wmma_store(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter, dim_n,
                       global_offset_C);
          }
        }
      }
    }
  }
}

#endif
