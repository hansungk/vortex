import numpy as np
# A_array = np.random.rand(16, 8)
# B_array = np.random.rand(8, 16)
A_array = np.zeros((16, 8))
B_array = np.zeros((8, 16))
A_array[0,:] = 1.0
B_array[:,0] = 1.0
C_array = np.random.rand(16, 16)


with open('a_matrix.h', 'w') as f:
    for i in range(A_array.shape[0]):
        for j in range(A_array.shape[1]):
            f.write(f'{A_array[i,j]}f, ')
        f.write('\n')

with open('b_matrix.h', 'w') as f:
    for i in range(B_array.shape[0]):
        for j in range(B_array.shape[1]):
            f.write(f'{B_array[i,j]}f, ')
        f.write('\n')

with open('c_matrix.h', 'w') as f:
    for i in range(C_array.shape[0]):
        for j in range(C_array.shape[1]):
            f.write(f'{C_array[i,j]}f, ')
        f.write('\n')

np.savez("abc", A_array=A_array, B_array=B_array, C_array=C_array)