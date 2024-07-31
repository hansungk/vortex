import numpy as np

# A_array = np.random.rand(8, 16)
A_array = np.arange(8 * 8).reshape([8, 8])
B_array = np.arange(8 * 8).reshape([8, 8])
# C_array = np.random.rand(16, 16)
C_array = np.zeros([8, 8])
# A_array = np.zeros((16, 8))
# B_array = np.zeros((8, 16))
# A_array[0,:] = 1.0
# B_array[:,4] = 1.0
# C_array = np.zeros((16, 16))
# for i in range(16):
#     for j in range(16):
#         C_array[i,j] = i * 16 + j

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
    result = T_packed.transpose([1, 0, 2]).reshape([rows // 2, cols * 2])
    return result


if __name__ == "__main__":
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

    A_array.astype('float32').tofile("input.a.bin")
    B_array.astype('float32').tofile("input.b.bin")

    # A_array.astype('float16').tofile("input.a.bin")
    # B_array = pack_fp16_by_column(B_array)
    # B_array.astype('float16').tofile("input.b.bin")
    print(B_array)
