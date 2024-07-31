#define RISCV_CUSTOM3   0x7B

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <stdio.h>
#include <vx_print.h>
#include "test_data.h"

constexpr int DIM_M = 8;

// single "substep" wmma instruction
// use accum buffer 0 (f16-f23)
inline void vx_wmma_acc0() {
	asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
}

// single "substep" wmma instruction
// use accum buffer 1 (f24-f31)
inline void vx_wmma_acc1() {
        asm volatile (".insn r %0, 0, 0, x1, x0, x0" :: "i"(RISCV_CUSTOM3));
}

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

void vx_wmma_load() {
  int tid = vx_thread_id();
  int tg = tid / 4;

  int row = 0;
  int col = 0;

  map_operand_8lanes(tid, row, col);

  // load A
  // each operand element is read twice by two threadgroups (Sec. III-B);
  // i.e. 8 regs * 32 lanes = 256 fp32 elements = 2 * (16 * 8) elements
  asm volatile("flw f0, %0" ::"m"(A[row][0]));
  asm volatile("flw f1, %0" ::"m"(A[row][1]));
  asm volatile("flw f2, %0" ::"m"(A[row][2]));
  asm volatile("flw f3, %0" ::"m"(A[row][3]));
  asm volatile("flw f4, %0" ::"m"(A[row][4]));
  asm volatile("flw f5, %0" ::"m"(A[row][5]));
  asm volatile("flw f6, %0" ::"m"(A[row][6]));
  asm volatile("flw f7, %0" ::"m"(A[row][7]));

  // load B
  asm volatile("flw f8 , %0" ::"m"(B[0][col]));
  asm volatile("flw f9 , %0" ::"m"(B[1][col]));
  asm volatile("flw f10, %0" ::"m"(B[2][col]));
  asm volatile("flw f11, %0" ::"m"(B[3][col]));
  asm volatile("flw f12, %0" ::"m"(B[4][col]));
  asm volatile("flw f13, %0" ::"m"(B[5][col]));
  asm volatile("flw f14, %0" ::"m"(B[6][col]));
  asm volatile("flw f15, %0" ::"m"(B[7][col]));

  map_c_8lanes(tid, row, col);

  // load C
  // accum buffer 0
  asm volatile("flw f16, %0" ::"m"(C[row + 0][col + 0]));
  asm volatile("flw f17, %0" ::"m"(C[row + 0][col + 1]));
  asm volatile("flw f18, %0" ::"m"(C[row + 2][col + 0]));
  asm volatile("flw f19, %0" ::"m"(C[row + 2][col + 1]));
  asm volatile("flw f20, %0" ::"m"(C[row + 0][col + 4]));
  asm volatile("flw f21, %0" ::"m"(C[row + 0][col + 5]));
  asm volatile("flw f22, %0" ::"m"(C[row + 2][col + 4]));
  asm volatile("flw f23, %0" ::"m"(C[row + 2][col + 5]));
  // accum buffer 1
  asm volatile("flw f24, %0" ::"m"(C[row + 0][col + 0]));
  asm volatile("flw f25, %0" ::"m"(C[row + 0][col + 1]));
  asm volatile("flw f26, %0" ::"m"(C[row + 2][col + 0]));
  asm volatile("flw f27, %0" ::"m"(C[row + 2][col + 1]));
  asm volatile("flw f28, %0" ::"m"(C[row + 0][col + 4]));
  asm volatile("flw f29, %0" ::"m"(C[row + 0][col + 5]));
  asm volatile("flw f30, %0" ::"m"(C[row + 2][col + 4]));
  asm volatile("flw f31, %0" ::"m"(C[row + 2][col + 5]));
}

// hardcoded device address for result
float *const results = reinterpret_cast<float *>(0xc0000000UL);

void store_wmma_result() {
  int wid = vx_warp_id();
  int tid = vx_thread_id();
  int tg = tid / 4;

  int row = 0;
  int col = 0;

  map_c_8lanes(tid, row, col);

  // store C
  float *const results_wid = results + (DIM_M * DIM_M * wid);
  // uncomment to have two accum buffers in rf
  // asm volatile("fsw f16, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 0)]));
  // asm volatile("fsw f17, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 1)]));
  // asm volatile("fsw f18, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 0)]));
  // asm volatile("fsw f19, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 1)]));
  // asm volatile("fsw f20, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 4)]));
  // asm volatile("fsw f21, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 5)]));
  // asm volatile("fsw f22, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 4)]));
  // asm volatile("fsw f23, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 5)]));
  asm volatile("fsw f24, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 0)]));
  asm volatile("fsw f25, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 1)]));
  asm volatile("fsw f26, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 0)]));
  asm volatile("fsw f27, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 1)]));
  asm volatile("fsw f28, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 4)]));
  asm volatile("fsw f29, %0" ::"m"(results_wid[DIM_M * (row + 0) + (col + 5)]));
  asm volatile("fsw f30, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 4)]));
  asm volatile("fsw f31, %0" ::"m"(results_wid[DIM_M * (row + 2) + (col + 5)]));
}

void print_wmma_result() {
  const int num_threads = vx_num_threads();

  for (int tid = 0; tid < num_threads; tid += 1) {
    for (int reg = 0; reg < 8; reg += 1) {
      vx_printf("thread %d, f%d: %x\n", tid, 16 + reg,
          *((int *)&results[tid * 8 + reg]));
    }
  }
}

void wmma() {
  vx_tmc(-1);

  // if (vx_warp_id() == 1) {
  //   for (int i = 0; i < 100; i++) {
  //     asm volatile ("nop");
  //   }
  // }

  vx_wmma_load();
  // #pragma GCC unroll 100
  // 	for (int i = 0; i < 100; i++) {
  // 		vx_wmma_acc0();
  // 	}
  vx_wmma_acc1();

  store_wmma_result();
  // print_wmma_result();
  vx_tmc(1);
}

int main() {
  const int num_warps = vx_num_warps();

  // vx_wspawn(num_warps, wmma);
  vx_wspawn(1, wmma);
  wmma();
  vx_wspawn_wait();

  return 0;
}
