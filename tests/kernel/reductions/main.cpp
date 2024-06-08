#define RISCV_CUSTOM2   0x5B
#define ADD_FUNC7       0b0000000
#define ADDU_FUNC7		0b1000000
#define MIN_FUNC7		0b0000001
#define MINU_FUNC7     	0b1000001
#define MAX_FUNC7 		0b0000010
#define MAXU_FUNC7		0b1000010
#define AND_FUNC7		0b0000011
#define OR_FUNC7		0b0000100
#define XOR_FUNC7		0b0000101

/*
 	6'h0: begin
		op_type = func7[6] ? `INST_RED_ADDU : `INST_RED_ADD;
	end
	6'h1: begin
		op_type = func7[6] ? `INST_RED_MINU : `INST_RED_MIN;
	end
	6'h2: begin
		op_type = func7[6] ? `INST_RED_MAXU : `INST_RED_MAX;
	end
	6'h3: begin
		op_type = `INST_RED_AND;
	end
	6'h4: begin
		op_type = `INST_RED_OR;
	end
	6'h5: begin
		op_type = `INST_RED_XOR;
	end
*/

#include <vx_intrinsics.h>
#include <stdio.h>
#include <vx_print.h>

int x[4] = {3, 7, 2, 5};
int y = -1;

inline int vx_add_reduce(int v) {
	int ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(ADD_FUNC7));
	return ret;
}

inline int vx_min_reduce(int v) {
	int ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(MIN_FUNC7));
	return ret;
}

inline unsigned vx_minu_reduce(unsigned v) {
	unsigned ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(MINU_FUNC7));
	return ret;
}

inline int vx_max_reduce(int v) {
	int ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(MAX_FUNC7));
	return ret;
}

inline unsigned vx_maxu_reduce(unsigned v) {
	unsigned ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(MAXU_FUNC7));
	return ret;
}


inline unsigned vx_and_reduce(unsigned v) {
	unsigned ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(AND_FUNC7));
	return ret;
}

inline unsigned vx_or_reduce(unsigned v) {
	unsigned ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(OR_FUNC7));
	return ret;
}

inline unsigned vx_xor_reduce(unsigned v) {
	unsigned ret;
	asm volatile (".insn r %2, 0, %3, %0, %1, x0" : "=r"(ret) : "r"(v), "i"(RISCV_CUSTOM2), "i"(XOR_FUNC7));
	return ret;
}

void test_add_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	int v = x[tid];
	int reduced = vx_add_reduce(v);
	vx_tmc(1);

	y = reduced;
}

unsigned unsigned_vector[4] = {(unsigned)-1, 0, (unsigned)-2, 5};

void test_min_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	int v = unsigned_vector[tid];
	int reduced = vx_min_reduce(v);
	vx_tmc(1);

	y = reduced;
}

void test_max_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	int v = unsigned_vector[tid];
	int reduced = vx_max_reduce(v);
	vx_tmc(1);

	y = reduced;
}

void test_minu_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	unsigned v = unsigned_vector[tid];
	unsigned reduced = vx_minu_reduce(v);
	vx_tmc(1);

	y = reduced;
}

void test_maxu_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	unsigned v = unsigned_vector[tid];
	unsigned reduced = vx_maxu_reduce(v);
	vx_tmc(1);

	y = reduced;
}

// assumes NUM_THREADS == 4
unsigned bit_vectors[4] = {0b11010110000111001100010100100110, 0b10010100011010001010000000001110, 0b10001001010111110001110000000010, 0b00010011010100101101110111001111};

void test_and_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	unsigned v = bit_vectors[tid];
	unsigned reduced = vx_and_reduce(v);
	vx_tmc(1);

	y = reduced;
}

void test_or_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	unsigned v = bit_vectors[tid];
	unsigned reduced = vx_or_reduce(v);
	vx_tmc(1);

	y = reduced;
}

void test_xor_reduce() {
	vx_tmc(-1);
	int tid = vx_thread_id();
	unsigned v = bit_vectors[tid];
	unsigned reduced = vx_xor_reduce(v);
	vx_tmc(1);

	y = reduced;
}

int main()
{
	int expected;

    test_add_reduce();
	vx_printf("add reduce result: %d\n", y);
	vx_printf("expected: %d\n", x[0] + x[1] + x[2] + x[3]);

	test_min_reduce();
	vx_printf("min reduce result: %d\n", y);
	expected = MIN((int)unsigned_vector[0], MIN((int)unsigned_vector[1], MIN((int)unsigned_vector[2], (int)unsigned_vector[3])));
	vx_printf("expected: %d\n", expected);

	test_max_reduce();
	vx_printf("max reduce result: %d\n", y);
	expected = MAX((int)unsigned_vector[0], MAX((int)unsigned_vector[1], MAX((int)unsigned_vector[2], (int)unsigned_vector[3])));
	vx_printf("expected: %d\n", expected);

	test_minu_reduce();
	vx_printf("minu reduce result: %d\n", y);
	expected = MIN(unsigned_vector[0], MIN(unsigned_vector[1], MIN(unsigned_vector[2], unsigned_vector[3])));
	vx_printf("expected: %d\n", expected);

	test_maxu_reduce();
	vx_printf("maxu reduce result: %d\n", y);
	expected = MAX(unsigned_vector[0], MAX(unsigned_vector[1], MAX(unsigned_vector[2], unsigned_vector[3])));
	vx_printf("expected: %d\n", expected);

	test_and_reduce();
	vx_printf("and reduce result: %d\n", y);
	vx_printf("expected: %d\n", bit_vectors[0] & bit_vectors[1] & bit_vectors[2] & bit_vectors[3]);


	test_or_reduce();
	vx_printf("or reduce result: %d\n", y);
	vx_printf("expected: %d\n", bit_vectors[0] | bit_vectors[1] | bit_vectors[2] | bit_vectors[3]);

	test_xor_reduce();
	vx_printf("xor reduce result: %d\n", y);
	vx_printf("expected: %d\n", bit_vectors[0] ^ bit_vectors[1] ^ bit_vectors[2] ^ bit_vectors[3]);
	

	return 0;
}
