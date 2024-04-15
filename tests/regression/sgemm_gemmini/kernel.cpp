#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define MATRIX_M 64  // TODO: remove hardcode
#define MATRIX_N 64
#define MATRIX_K 64
#define TILE_M 32        // tile size = SMEM size / 2 (double buffering) / 4 (A, B, C, Psum)
#define TILE_N 32
#define TILE_K 32
#define TILE_MN 1024
#define TILE_MK 1024
#define TILE_NK 1024

#define NUM_CLUSTERS 1
#define TB_M (MATRIX_M / NUM_CLUSTERS)
#define TB_N MATRIX_N
#define TB_SIZE (TB_M * TB_N)
#define NUM_TILE_ROWS_PER_TB (TB_M / TILE_M)
#define THREAD_ELEMS 8 // elements per thread in a tile
#define THREAD_STRIDE 8 // threads per core

#define SMEM_ADDR_0K ((float *) 0xff000000)
#define SMEM_ADDR_4K ((float *) 0xff001000)
#define SMEM_ADDR_8K ((float *) 0xff002000)
#define SMEM_ADDR_12K ((float *) 0xff003000)

#define SPAD_ADDR_0K 0x0
#define SPAD_ADDR_4K 0x80
#define SPAD_ADDR_8K 0x100
#define SPAD_ADDR_12K 0x180

// #define DEBUG_PRINT
#define rd_cycles(x) asm volatile ("csrr %0, mcycle" : "=r" (x))

void threadblock_barrier(unsigned int tid_in_threadblock, unsigned int barrier_id, unsigned int count) {
  vx_fence();
  vx_barrier(barrier_id, count);
}

void thread_block_matmul_gemmini(kernel_arg_t *__UNIFORM__ arg,
                                 const uint32_t threadblock_id,
                                 const uint32_t tid_in_threadblock) {
  const float * const A = (const float * const) arg->addr_a;
  const float * const B = (const float * const) arg->addr_b;
  float * const C = (float * const) arg->addr_c;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;
  const uint32_t num_tiles_n = dim_n / TILE_N;
  const uint32_t num_tiles_k = dim_k / TILE_K;
  // TODO: make this into constexpr by subbing architectural params with macros
  const uint32_t num_threads_in_cluster = vx_num_threads() * vx_num_warps() * CORES_PER_CLUSTER;
  const uint32_t hw_tid = tid_in_threadblock % num_threads_in_cluster;
  const uint32_t a_elems_per_thread = TILE_MK / num_threads_in_cluster;
  const uint32_t b_elems_per_thread = TILE_NK / num_threads_in_cluster;
  const uint32_t c_elems_per_thread = TILE_MN / num_threads_in_cluster;
  const uint32_t thread_load_offset = hw_tid;
  const uint32_t thread_load_stride = num_threads_in_cluster;

  uint32_t marker0, marker1, marker2, marker3, marker4;
  uint32_t marker5, marker6, marker7, marker8, marker9;

  if (hw_tid == 0) {
    gemmini_config_ld(0);
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    gemmini_config_st(0);
    sprintf(PRINT_BUF, "start\n");
  }


  // TODO: check for tb id
  rd_cycles(marker0);

  for (int tile_i = NUM_TILE_ROWS_PER_TB * threadblock_id;
           tile_i < NUM_TILE_ROWS_PER_TB * (threadblock_id + 1);
           tile_i += 1) {
    for (int tile_j = 0; tile_j < num_tiles_n; tile_j += 1) {
      float * const smem_c_tile_start = SMEM_ADDR_4K;
      float * const dram_c_tile_start = C + tile_i * TILE_M * dim_n + tile_j * TILE_N;

      for (int tile_k = 0; tile_k < num_tiles_k; tile_k += 1) {
        // TODO: double buffer
        const float * const dram_a_tile_start = A + tile_i * TILE_M * dim_k + tile_k * TILE_K;
        const float * const dram_b_tile_start = B + tile_k * TILE_K * dim_n + tile_j * TILE_N;
        float * const smem_a_tile_start = SMEM_ADDR_0K;
        float * const smem_b_tile_start = SMEM_ADDR_12K;

        rd_cycles(marker1);

        // preload A matrix
#pragma GCC unroll 8 // TODO: macro computed
        for (int thread_i = 0; thread_i < a_elems_per_thread; thread_i++) {
          uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
          smem_a_tile_start[SMEM_MAT_OFFSET(elem_offset / TILE_K, elem_offset % TILE_K, TILE_K)] = \
            dram_a_tile_start[elem_offset / TILE_K * dim_k + elem_offset % TILE_K];
        }

#ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          sprintf(PRINT_BUF, "\nA %d %d\n", tile_i, tile_k);
          for (int i = 0; i < TILE_M; i += 8) {
            for (int j = 0; j < TILE_K; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_K);
              sprintf(PRINT_BUF, "%x %x ",
                (int) (smem_a_tile_start[mat_offset]),
                (int) (smem_a_tile_start[mat_offset + 4])
              );
            }
            sprintf(PRINT_BUF, "\n");
           }
        }
#endif

        // preload B matrix
#pragma GCC unroll 8
        for (int thread_i = 0; thread_i < b_elems_per_thread; thread_i++) {
          uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
          smem_b_tile_start[SMEM_MAT_OFFSET(elem_offset / TILE_N, elem_offset % TILE_N, TILE_N)] = \
            dram_b_tile_start[elem_offset / TILE_N * dim_n + elem_offset % TILE_N];
        }

#ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          sprintf(PRINT_BUF, "\nB %d %d\n", tile_k, tile_j);
          for (int i = 0; i < TILE_K; i += 8) {
            for (int j = 0; j < TILE_N; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
              sprintf(PRINT_BUF, "%x %x ",
                (int) (smem_b_tile_start[mat_offset]),
                (int) (smem_b_tile_start[mat_offset + 4])
              );
            }
            sprintf(PRINT_BUF, "\n");
          }
        }
#endif
        rd_cycles(marker2);

        // cluster wide barrier to wait for A and B loads to complete
        threadblock_barrier(0, /*barrier_id=*/threadblock_id, /*count=*/num_threads_in_cluster);
        rd_cycles(marker3);
        if (hw_tid == 0) {
          sp_tiled_matmul_full_spad_ws(SPAD_ADDR_0K, SPAD_ADDR_12K, /*spad_D=*/0, SPAD_ADDR_4K,
            /*I=*/TILE_M / DIM, /*J=*/TILE_N / DIM, /*K=*/TILE_K / DIM, /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*no_bias=*/1, /*repeating_bias=*/0, /*act=*/NO_ACTIVATION);
          gemmini_fence();
        }
        rd_cycles(marker4);
        threadblock_barrier(0, /*barrier_id=*/threadblock_id, /*count=*/num_threads_in_cluster);
        rd_cycles(marker5);

        // accumulate C matrix
        if (tile_k == 0) {
#pragma GCC unroll 8
          for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
            uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
            *(SMEM_ADDR_8K + elem_offset) = smem_c_tile_start[elem_offset];
          }
        } else {
#pragma GCC unroll 8
          for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
            uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
            *(SMEM_ADDR_8K + elem_offset) += smem_c_tile_start[elem_offset];
          }
        }

        rd_cycles(marker6);
#ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          sprintf(PRINT_BUF, "\nC %d %d %d\n", tile_i, tile_j, tile_k);
          for (int i = 0; i < TILE_M; i += 8) {
            for (int j = 0; j < TILE_N; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
              sprintf(PRINT_BUF, "%d %d ",
                (int) (smem_c_tile_start[mat_offset]),
                (int) (smem_c_tile_start[mat_offset + 4])
              );
            }
            sprintf(PRINT_BUF, "\n");
          }
        }
#endif
      }

      rd_cycles(marker7);
      // move out to dram
 #pragma GCC unroll 8 // TODO: macro computed
      for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
        uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
        dram_c_tile_start[elem_offset / TILE_N * dim_n + elem_offset % TILE_N] = \
          *(SMEM_ADDR_8K + SMEM_MAT_OFFSET(elem_offset / TILE_N, elem_offset % TILE_N, TILE_N));
      }

      rd_cycles(marker8);
      /* if (hw_tid == 0) {
        sprintf(PRINT_BUF, "\nC %d %d\n", tile_i, tile_j);
        for (int i = 0; i < TILE_M; i += 8) {
          for (int j = 0; j < TILE_N; j += 8) {
            uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
            sprintf(PRINT_BUF, "%d %d ",
              (int) (C[(tile_i * TILE_M + i) * dim_n + tile_j * TILE_N + j]),
              (int) (C[(tile_i * TILE_M + i) * dim_n + tile_j * TILE_N + j + 4])
            );
          }
          sprintf(PRINT_BUF, "\n");
        }
      } */
    }
  }
  // last thread block complete
  if (threadblock_id == NUM_CLUSTERS - 1) {
    threadblock_barrier(0, /*barrier_id=*/0, /*count=*/num_threads_in_cluster);
    rd_cycles(marker9);
    if (hw_tid == 0) {
      sprintf(PRINT_BUF, "complete\n");
      sprintf(PRINT_BUF, "total cycles:         %d\n", marker9 - marker0);
      sprintf(PRINT_BUF, "single tile cycles:   %d\n", marker6 - marker1);
      sprintf(PRINT_BUF, "A/B tile load cycles: %d\n", marker2 - marker1);
      sprintf(PRINT_BUF, "gemmini cycles:       %d\n", marker4 - marker3);
      sprintf(PRINT_BUF, "first barrier:        %d\n", marker3 - marker2);
      sprintf(PRINT_BUF, "second barrier:       %d\n", marker5 - marker4);
      sprintf(PRINT_BUF, "accumulation cycles:  %d\n", marker6 - marker5);
      sprintf(PRINT_BUF, "dram mvout cycles:    %d\n", marker8 - marker7);
    }
    threadblock_barrier(0, /*barrier_id=*/0, /*count=*/num_threads_in_cluster);
    if (hw_tid == num_threads_in_cluster - 1) {
      sprintf(PRINT_BUF, "single tile cycles:   %d\n", marker6 - marker1);
      sprintf(PRINT_BUF, "A/B tile load cycles: %d\n", marker2 - marker1);
      sprintf(PRINT_BUF, "gemmini cycles:       %d\n", marker4 - marker3);
      sprintf(PRINT_BUF, "first barrier:        %d\n", marker3 - marker2);
      sprintf(PRINT_BUF, "second barrier:       %d\n", marker5 - marker4);
    }
    vx_tmc_one();
  }
  vx_tmc(0);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

  const int threadblock_id = task_id / TB_SIZE;
  const int tid_in_threadblock = task_id % TB_SIZE;

  thread_block_matmul_gemmini(arg, threadblock_id, tid_in_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  sprintf(PRINT_BUF, "m=%d, n=%d\n", arg->dim_m, arg->dim_n);

  const uint32_t num_threads_in_cluster = vx_num_threads() * vx_num_warps() * CORES_PER_CLUSTER;
  const uint32_t grid_size = num_threads_in_cluster * NUM_CLUSTERS;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}