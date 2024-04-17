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

#define SMEM_ADDR_0K  ((float * const) 0xff000000)
#define SMEM_ADDR_4K  ((float * const) 0xff001000)
#define SMEM_ADDR_8K  ((float * const) 0xff002000)
#define SMEM_ADDR_12K ((float * const) 0xff003000)

#define SPAD_ADDR_0K 0x0
#define SPAD_ADDR_4K 0x80
#define SPAD_ADDR_8K 0x100
#define SPAD_ADDR_12K 0x180

#define HARDCODE
#define PRINTF(...) sprintf(PRINT_BUF, __VA_ARGS__)
//#define PRINTF(...) vx_printf(__VA_ARGS__)

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
  // const uint32_t num_threads_in_cluster = vx_num_threads() * vx_num_warps() * CORES_PER_CLUSTER;
  constexpr uint32_t num_threads_in_cluster = 128;
  constexpr uint32_t a_elems_per_thread = TILE_MK / num_threads_in_cluster;
  constexpr uint32_t b_elems_per_thread = TILE_NK / num_threads_in_cluster;
  constexpr uint32_t c_elems_per_thread = TILE_MN / num_threads_in_cluster;
  const uint32_t hw_tid = tid_in_threadblock % num_threads_in_cluster;
  const uint32_t thread_load_offset = hw_tid;
  constexpr uint32_t thread_load_stride = num_threads_in_cluster;

  // the dram coordinates are (i1 + i0, j1 + j0). i0 and j0 are both spatially mapped only.
  const uint32_t j0 = hw_tid % DIM;
  const uint32_t i0 = (hw_tid / DIM) % DIM;

  // j1 is both spatially and temporally mapped. j1 increases every iteration.
  const uint32_t j1_idx = (hw_tid / DIM / DIM) * DIM; // A: % TILE_K, B: % TILE_N, C: % TILE_N
  // every iteratioon, j1 increases by j1_stride
  constexpr uint32_t j1_stride = (num_threads_in_cluster / DIM / DIM) * DIM; // mod TILE_W after stride

  // i1 is only temporally mapped. i1 increments every one or more iterations
  constexpr uint32_t i1_stride = DIM; // step per increment (increment doesnt happen every iteration)
  constexpr uint32_t i1_iters = (DIM * DIM * (TILE_K / DIM)) / num_threads_in_cluster; // num of iters before striding

  uint32_t marker0, marker1, marker2, marker3, marker4;
  uint32_t marker5, marker6, marker7, marker8, marker9;

  if (hw_tid == 0) {
    gemmini_config_ld(0);
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    gemmini_config_st(0);
    PRINTF("start\n");
  }

  // TODO: check for tb id
  rd_cycles(marker0);

  for (int tile_i = NUM_TILE_ROWS_PER_TB * threadblock_id;
           tile_i < NUM_TILE_ROWS_PER_TB * (threadblock_id + 1);
           tile_i += 1) {
    for (int tile_j = 0; tile_j < num_tiles_n; tile_j += 1) {
      float * const smem_c_tile_start = SMEM_ADDR_4K;
      float * const smem_acc_tile_start = SMEM_ADDR_8K;
      float * const dram_c_tile_start = C + tile_i * TILE_M * dim_n + tile_j * TILE_N;

      for (int tile_k = 0; tile_k < num_tiles_k; tile_k += 1) {
        // TODO: double buffer
        rd_cycles(marker1);

        #ifdef HARDCODE
          #if (TILE_MK / NUM_THREADS / NUM_WARPS / CORES_PER_CLUSTER) != 8
            #error CANNOT UNROLL
          #endif
        // preload A B matrix

        constexpr uint32_t every_iter = j1_stride;
        const uint32_t every_2iters_a = i1_stride * dim_k;
        const uint32_t runtime_const_a = i0 * dim_k + j1_idx + j0;
        const uint32_t every_2iters_b = i1_stride * dim_n;
        const uint32_t runtime_const_b = i0 * dim_n + j1_idx + j0;

        const float * const dram_a_tile_start = A + tile_i * TILE_M * dim_k + tile_k * TILE_K + runtime_const_a;
        const float * const dram_b_tile_start = B + tile_k * TILE_K * dim_n + tile_j * TILE_N + runtime_const_b;
        float * const smem_a_tile_start = SMEM_ADDR_0K + hw_tid;
        float * const smem_b_tile_start = SMEM_ADDR_12K + hw_tid;

        const float v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 0];
        const float w0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 0];
        const float v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 0];
        const float w1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 0];
        const float v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 1];
        const float w2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 1];
        const float v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 1];
        const float w3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 1];
        const float v4 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 2];
        const float w4 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 2];
        const float v5 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 2];
        const float w5 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 2];
        const float v6 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 3];
        const float w6 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 3];
        const float v7 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 3];
        const float w7 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 3];

        smem_a_tile_start[0 * num_threads_in_cluster] = v0;
        smem_b_tile_start[0 * num_threads_in_cluster] = w0;
        smem_a_tile_start[1 * num_threads_in_cluster] = v1;
        smem_b_tile_start[1 * num_threads_in_cluster] = w1;
        smem_a_tile_start[2 * num_threads_in_cluster] = v2;
        smem_b_tile_start[2 * num_threads_in_cluster] = w2;
        smem_a_tile_start[3 * num_threads_in_cluster] = v3;
        smem_b_tile_start[3 * num_threads_in_cluster] = w3;
        smem_a_tile_start[4 * num_threads_in_cluster] = v4;
        smem_b_tile_start[4 * num_threads_in_cluster] = w4;
        smem_a_tile_start[5 * num_threads_in_cluster] = v5;
        smem_b_tile_start[5 * num_threads_in_cluster] = w5;
        smem_a_tile_start[6 * num_threads_in_cluster] = v6;
        smem_b_tile_start[6 * num_threads_in_cluster] = w6;
        smem_a_tile_start[7 * num_threads_in_cluster] = v7;
        smem_b_tile_start[7 * num_threads_in_cluster] = w7;

        /* smem_a_tile_start[0 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 0 + every_2iters * 0];
        smem_a_tile_start[1 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 1 + every_2iters * 0];
        smem_a_tile_start[2 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 0 + every_2iters * 1];
        smem_a_tile_start[3 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 1 + every_2iters * 1];
        smem_a_tile_start[4 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 0 + every_2iters * 2];
        smem_a_tile_start[5 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 1 + every_2iters * 2];
        smem_a_tile_start[6 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 0 + every_2iters * 3];
        smem_a_tile_start[7 * num_threads_in_cluster + hw_tid] = \
          dram_a_tile_start[runtime_const + every_iter * 1 + every_2iters * 3];

        smem_b_tile_start[0 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 0 + every_2iters * 0];
        smem_b_tile_start[1 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 1 + every_2iters * 0];
        smem_b_tile_start[2 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 0 + every_2iters * 1];
        smem_b_tile_start[3 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 1 + every_2iters * 1];
        smem_b_tile_start[4 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 0 + every_2iters * 2];
        smem_b_tile_start[5 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 1 + every_2iters * 2];
        smem_b_tile_start[6 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 0 + every_2iters * 3];
        smem_b_tile_start[7 * num_threads_in_cluster + hw_tid] = \
          dram_b_tile_start[runtime_const + every_iter * 1 + every_2iters * 3]; */
        #else
        const float * const dram_a_tile_start = A + tile_i * TILE_M * dim_k + tile_k * TILE_K;
        const float * const dram_b_tile_start = B + tile_k * TILE_K * dim_n + tile_j * TILE_N;
        float * const smem_a_tile_start = SMEM_ADDR_0K;
        float * const smem_b_tile_start = SMEM_ADDR_12K;

        #pragma GCC unroll 8 // TODO: macro computed
        for (uint32_t thread_i = 0, j1 = 0, i1 = 0;
          thread_i < a_elems_per_thread;
          thread_i += 1,
          j1 = (j1 + j1_stride) % TILE_K,
          i1 = (thread_i % i1_iters == 0) ? i1 + i1_stride : i1) {
          smem_a_tile_start[thread_i * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[(0 + i0) * dim_k + j1 + j1_idx + j0];
        }
        // for (int thread_i = 0; thread_i < a_elems_per_thread; thread_i++) {
        //   uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
        //   smem_a_tile_start[SMEM_MAT_OFFSET(elem_offset / TILE_K, elem_offset % TILE_K, TILE_K)] = \
        //     dram_a_tile_start[elem_offset / TILE_K * dim_k + elem_offset % TILE_K];
        // }
        #pragma GCC unroll 8
        for (int thread_i = 0; thread_i < b_elems_per_thread; thread_i++) {
          uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
          smem_b_tile_start[SMEM_MAT_OFFSET(elem_offset / TILE_N, elem_offset % TILE_N, TILE_N)] = \
            dram_b_tile_start[elem_offset / TILE_N * dim_n + elem_offset % TILE_N];
        }
        #endif

        #ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          PRINTF("\nA %d %d\n", tile_i, tile_k);
          for (int i = 0; i < TILE_M; i += 8) {
            for (int j = 0; j < TILE_K; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_K);
              PRINTF("%x %x ",
                (int) (smem_a_tile_start[mat_offset]),
                (int) (smem_a_tile_start[mat_offset + 4])
              );
            }
            PRINTF("\n");
          }
          PRINTF("\nB %d %d\n", tile_k, tile_j);
          for (int i = 0; i < TILE_K; i += 8) {
            for (int j = 0; j < TILE_N; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
              PRINTF("%x %x ",
                (int) (smem_b_tile_start[mat_offset]),
                (int) (smem_b_tile_start[mat_offset + 4])
              );
            }
            PRINTF("\n");
          }
        }
        #endif


        rd_cycles(marker2);
        // cluster wide barrier to wait for A and B loads to complete
        threadblock_barrier(0, /*barrier_id=*/threadblock_id, /*count=*/NUM_WARPS);
        rd_cycles(marker3);
        if (hw_tid == 0) {
          sp_tiled_matmul_full_spad_ws(SPAD_ADDR_0K, SPAD_ADDR_12K, /*spad_D=*/0, SPAD_ADDR_4K,
            /*I=*/TILE_M / DIM, /*J=*/TILE_N / DIM, /*K=*/TILE_K / DIM, /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*no_bias=*/1, /*repeating_bias=*/0, /*act=*/NO_ACTIVATION);
          gemmini_fence();
        }
        rd_cycles(marker4);
        threadblock_barrier(0, /*barrier_id=*/threadblock_id, /*count=*/NUM_WARPS);
        rd_cycles(marker5);

        // accumulate C matrix
        if (tile_k == 0) {
          #pragma GCC ivdep
          #pragma GCC unroll 8
          for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
            uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
            smem_acc_tile_start[elem_offset] = smem_c_tile_start[elem_offset];
          }
        } else {
          #if (TILE_NK / NUM_THREADS / NUM_WARPS / CORES_PER_CLUSTER) != 8
          #error CANNOT UNROLL
          #endif
          for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i += 8) {
            constexpr uint32_t s = num_threads_in_cluster;
            smem_acc_tile_start[hw_tid + s * 0] += smem_c_tile_start[hw_tid + s * 0];
            smem_acc_tile_start[hw_tid + s * 1] += smem_c_tile_start[hw_tid + s * 1];
            smem_acc_tile_start[hw_tid + s * 2] += smem_c_tile_start[hw_tid + s * 2];
            smem_acc_tile_start[hw_tid + s * 3] += smem_c_tile_start[hw_tid + s * 3];
            smem_acc_tile_start[hw_tid + s * 4] += smem_c_tile_start[hw_tid + s * 4];
            smem_acc_tile_start[hw_tid + s * 5] += smem_c_tile_start[hw_tid + s * 5];
            smem_acc_tile_start[hw_tid + s * 6] += smem_c_tile_start[hw_tid + s * 6];
            smem_acc_tile_start[hw_tid + s * 7] += smem_c_tile_start[hw_tid + s * 7];
          }
        }

        rd_cycles(marker6);
        #ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          PRINTF("\nC %d %d %d\n", tile_i, tile_j, tile_k);
          for (int i = 0; i < TILE_M; i += 8) {
            for (int j = 0; j < TILE_N; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
              PRINTF("%d %d ",
                (int) (smem_c_tile_start[mat_offset]),
                (int) (smem_c_tile_start[mat_offset + 4])
              );
            }
            PRINTF("\n");
          }
        }
        #endif
      }

      rd_cycles(marker7);
      // move out to dram

      #ifdef HARDCODE
      #if (TILE_MN / NUM_THREADS / NUM_WARPS / CORES_PER_CLUSTER) != 8
        #error CANNOT UNROLL
      #endif
      constexpr uint32_t every_iter = j1_stride;
      const uint32_t every_2iters = i1_stride * dim_n;
      const uint32_t runtime_const = i0 * dim_n + j1_idx + j0;
      dram_c_tile_start[runtime_const + every_iter * 0 + every_2iters * 0] = \
        smem_acc_tile_start[0 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 1 + every_2iters * 0] = \
        smem_acc_tile_start[1 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 0 + every_2iters * 1] = \
        smem_acc_tile_start[2 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 1 + every_2iters * 1] = \
        smem_acc_tile_start[3 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 0 + every_2iters * 2] = \
        smem_acc_tile_start[4 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 1 + every_2iters * 2] = \
        smem_acc_tile_start[5 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 0 + every_2iters * 3] = \
        smem_acc_tile_start[6 * num_threads_in_cluster + hw_tid];
      dram_c_tile_start[runtime_const + every_iter * 1 + every_2iters * 3] = \
        smem_acc_tile_start[7 * num_threads_in_cluster + hw_tid];
      #else
      #pragma GCC unroll 8
      for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
        uint32_t elem_offset = thread_load_offset + thread_load_stride * thread_i;
        dram_c_tile_start[elem_offset / TILE_N * dim_n + elem_offset % TILE_N] = \
          *(SMEM_ADDR_8K + SMEM_MAT_OFFSET(elem_offset / TILE_N, elem_offset % TILE_N, TILE_N));
      }
      #endif

      rd_cycles(marker8);
      /* if (hw_tid == 0) {
        sprintf(PRINT_BUF, "\nC %d %d\n", tile_i, tile_j);
        for (int i = 0; i < TILE_M; i += 8) {
          for (int j = 0; j < TILE_N; j += 8) {
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
    threadblock_barrier(0, /*barrier_id=*/0, /*count=*/NUM_WARPS);
    rd_cycles(marker9);
    if (hw_tid == 0) {
      PRINTF("\ncomplete\n");
      PRINTF("total cycles:         %d\n", marker9 - marker0);
      PRINTF("tile start:           %d\n", marker1);
      PRINTF("single tile cycles:   %d\n", marker6 - marker1);
      PRINTF("A/B tile load cycles: %d\n", marker2 - marker1);
      PRINTF("first barrier:        %d\n", marker3 - marker2);
      PRINTF("gemmini cycles:       %d\n", marker4 - marker3);
      PRINTF("second barrier:       %d\n", marker5 - marker4);
      PRINTF("accumulation cycles:  %d\n", marker6 - marker5);
      PRINTF("dram mvout cycles:    %d\n", marker8 - marker7);
    }
    threadblock_barrier(0, /*barrier_id=*/1, /*count=*/NUM_WARPS);
    if (hw_tid == num_threads_in_cluster - 1) {
      PRINTF("\ntile start:           %d\n", marker1);
      PRINTF("single tile cycles:   %d\n", marker6 - marker1);
      PRINTF("A/B tile load cycles: %d\n", marker2 - marker1);
      PRINTF("first barrier:        %d\n", marker3 - marker2);
      PRINTF("gemmini cycles:       %d\n", marker4 - marker3);
      PRINTF("second barrier:       %d\n", marker5 - marker4);
      PRINTF("accumulation cycles:  %d\n", marker6 - marker5);
      PRINTF("dram mvout cycles:    %d\n", marker8 - marker7);
    }
    threadblock_barrier(0, /*barrier_id=*/2, /*count=*/NUM_WARPS);
    if (hw_tid == 0) {
      for (int i = 0; i < dim_m; i += 8) {
        for (int j = 0; j < dim_n; j += 8) {
          sprintf(PRINT_BUF, "%d %d ",
                  (int) (C[i * dim_n + j]),
                  (int) (C[i * dim_n + j + 4])
          );
        }
        PRINTF("\n");
      }
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