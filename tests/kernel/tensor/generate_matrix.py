import sys
import numpy as np

def parse_mnk():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} dimM dimN dimK", file=sys.stderr)
        sys.exit(1)
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    return (m, n, k)


# Reorder array in a way that groups two adjacent elements along the column to
# be now adjacent along the row.  This way, when the resulting fp16 array is
# read in column-major order with 32-bit granularity, the fp16 elements will be
# read in the same order as regular fp32 elements in column-major.
#
# For example:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# becomes
# [[1 3 2 4]
#  [5 7 6 8]]
def pack_fp16_by_column(array):
    rows = array.shape[0]
    cols = array.shape[1]

    T = array.transpose([1, 0])
    T_packed = T.reshape([cols, -1, 2])
    result = T_packed.transpose([1, 0, 2])
    return result


# Do the same as pack_fp16_by_column, but for every two elements along the row.
def pack_fp16_by_row(array):
    rows = array.shape[0]
    cols = array.shape[1]

    result = array.reshape([rows, -1, 2])
    return result


if __name__ == "__main__":
    M, N, K = parse_mnk()

    rand = True
    if not rand:
        A_array = np.arange(M * K).reshape([M, K])
        B_array = np.arange(K * N).reshape([K, N])
        C_array = np.arange(M * N).reshape([M, N])
    else:
        np.random.seed(0)
        A_array = np.random.rand(M, K) - 0.5
        B_array = np.random.rand(K, N) - 0.5
        C_array = np.random.rand(N, K) - 0.5
        # C_array = np.zeros([M, N])

    with open('a_matrix.h', 'w') as f:
        for i in range(A_array.shape[0]):
            for j in range(A_array.shape[1]):
                f.write(f'{A_array[i,j]:f}f, ')
            f.write('\n')
    with open('b_matrix.h', 'w') as f:
        for i in range(B_array.shape[0]):
            for j in range(B_array.shape[1]):
                f.write(f'{B_array[i,j]:f}f, ')
            f.write('\n')
    with open('c_matrix.h', 'w') as f:
        for i in range(C_array.shape[0]):
            for j in range(C_array.shape[1]):
                f.write(f'{C_array[i,j]:f}f, ')
            f.write('\n')

    np.savez("abc", A_array=A_array, B_array=B_array, C_array=C_array)

    D_expected = A_array @ B_array
    D_expected.astype('float32').tofile("d_expected.bin")
    print('D_expected:')
    print(D_expected)

    fp16 = False
    if fp16:
        A_packed = pack_fp16_by_row(A_array)
        AT_packed = A_packed.transpose([1, 0, 2])
        AT_array = AT_packed.reshape([-1, M * 2])
        AT_array.astype('float16').tofile("input.a.col.bin")
        print('AT:')
        print(AT_array)
        B_packed = pack_fp16_by_column(B_array)
        B_array = B_packed.reshape([-1, N * 2])
        B_array.astype('float16').tofile("input.b.row.bin")
        print('B:')
        print(B_array)
    else:
        A_array.astype('float32').tofile("input.a.row.bin")
        AT_array = A_array.transpose([1, 0])
        AT_array.astype('float32').tofile("input.a.col.bin")
        B_array.astype('float32').tofile("input.b.bin")
        C_array.astype('float32').tofile("input.c.bin")
        print('AT:')
        print(AT_array)
        print('B:')
        print(B_array)

    # generate rowmax result in online softmax
    row_max = np.max(D_expected, axis=1)
    row_max.astype('float32').tofile("rowmax.bin")

    # subtrace row_max from each row by broadcasting
    # (placeholder for exp)
    x = D_expected - row_max[:, np.newaxis]
    P = (x**2) / 2.0 + x + 1.0
    # for i in range(3, 4):
    #     P += (x**i) / np.math.factorial(i)
    # P = np.exp(exp)
    P.astype('float32').tofile("P_expected.bin")
    # print('P error:')
    # print(P / np.exp(x))
    print('P_expected:')
    print(P)

    row_sum = np.sum(P, axis=1)
    row_sum.astype('float32').tofile("rowsum.bin")

    V = C_array
    # O = P.transpose([1, 0]) @ V
    O = P @ V
    O.astype('float32').tofile("O_expected.bin")
    print('O_expected:')
    print(O)
