#define RISCV_CUSTOM3   0x7B

#include <vx_intrinsics.h>
#include <stdio.h>
#include <vx_print.h>

inline void vx_wmma() {
	asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
}

#include "test_data.h"

void vx_wmma_load() {
	int tid = vx_thread_id();
	int tg = tid / 4;

	// load A
	int row = tid % 4;
	row += (tg * 8) % 16;
	row += (tg / 4) * 4;

	asm volatile ("flw f0, %0" :: "m"(A[row][0])); 
	asm volatile ("flw f1, %0" :: "m"(A[row][1])); 
	asm volatile ("flw f2, %0" :: "m"(A[row][2])); 
	asm volatile ("flw f3, %0" :: "m"(A[row][3])); 
	asm volatile ("flw f4, %0" :: "m"(A[row][4])); 
	asm volatile ("flw f5, %0" :: "m"(A[row][5])); 
	asm volatile ("flw f6, %0" :: "m"(A[row][6])); 
	asm volatile ("flw f7, %0" :: "m"(A[row][7])); 

	// load B
	int col = tid % 4;
	col += ((tg % 4) / 2) * 8;
	col += (tg / 4) * 4;

	asm volatile ("flw f8 , %0" :: "m"(B[0][col])); 
	asm volatile ("flw f9 , %0" :: "m"(B[1][col])); 
	asm volatile ("flw f10, %0" :: "m"(B[2][col])); 
	asm volatile ("flw f11, %0" :: "m"(B[3][col])); 
	asm volatile ("flw f12, %0" :: "m"(B[4][col])); 
	asm volatile ("flw f13, %0" :: "m"(B[5][col])); 
	asm volatile ("flw f14, %0" :: "m"(B[6][col])); 
	asm volatile ("flw f15, %0" :: "m"(B[7][col])); 

	// load C
	col = ((tg % 4) / 2) * 8;
	row = (tg * 8) % 16;
	row += (tg / 4) * 4;

	row += (tid % 4) % 2;
    col += ((tid % 4) / 2) * 2;

	asm volatile ("flw f16, %0" :: "m"(C[row+0][col+0])); 
	asm volatile ("flw f17, %0" :: "m"(C[row+0][col+1])); 
	asm volatile ("flw f18, %0" :: "m"(C[row+2][col+0])); 
	asm volatile ("flw f19, %0" :: "m"(C[row+2][col+1])); 
	asm volatile ("flw f20, %0" :: "m"(C[row+0][col+4])); 
	asm volatile ("flw f21, %0" :: "m"(C[row+0][col+5])); 
	asm volatile ("flw f22, %0" :: "m"(C[row+2][col+4])); 
	asm volatile ("flw f23, %0" :: "m"(C[row+2][col+5])); 
}

float results[32*8];

void store_wmma_result() {
	int tid = vx_thread_id();
	
	float *results = reinterpret_cast<float *>(0xc0000000UL);
	asm volatile ("fsw f16, %0" :: "m"(results[tid*8+0])); 
	asm volatile ("fsw f17, %0" :: "m"(results[tid*8+1])); 
	asm volatile ("fsw f18, %0" :: "m"(results[tid*8+2])); 
	asm volatile ("fsw f19, %0" :: "m"(results[tid*8+3])); 
	asm volatile ("fsw f20, %0" :: "m"(results[tid*8+4])); 
	asm volatile ("fsw f21, %0" :: "m"(results[tid*8+5])); 
	asm volatile ("fsw f22, %0" :: "m"(results[tid*8+6])); 
	asm volatile ("fsw f23, %0" :: "m"(results[tid*8+7])); 
}

void print_wmma_result() {
	for (int tid = 0; tid < 32; tid += 1) {
		for (int reg = 0; reg < 8; reg += 1) {
			vx_printf("thread %d, f%d: %x\n", tid, 16+reg, *((int*) &results[tid*8+reg]));
		}
	}
}

int main()
{
	vx_tmc(-1);
	vx_wmma_load();
#pragma GCC unroll 100
	for (int i = 0; i < 100; i++) {
		vx_wmma();
	}
	store_wmma_result();
	vx_tmc(1);
	// print_wmma_result();
	
	return 0;
}
