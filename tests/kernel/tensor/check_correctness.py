import numpy as np
import struct

def hex2float(float_hex_str):
    # print(float_hex_str.strip())
    return struct.unpack(">f",struct.pack(">i",int(float_hex_str,16)))[0]


def C_index(threadgroup, thread, register):
    """
    col = ((tg % 4) / 2) * 8;
	row = (tg * 8) % 16;
	row += (tg / 4) * 4;

	asm volatile ("flw f16, %0" :: "m"(C[row+0][col+0])); 
	asm volatile ("flw f17, %0" :: "m"(C[row+0][col+1])); 
	asm volatile ("flw f18, %0" :: "m"(C[row+2][col+0])); 
	asm volatile ("flw f19, %0" :: "m"(C[row+2][col+1])); 
	asm volatile ("flw f20, %0" :: "m"(C[row+0][col+4])); 
	asm volatile ("flw f21, %0" :: "m"(C[row+0][col+5])); 
	asm volatile ("flw f22, %0" :: "m"(C[row+2][col+4])); 
	asm volatile ("flw f23, %0" :: "m"(C[row+2][col+5])); 
    """

    col = ((threadgroup % 4) // 2) * 8
    row = (threadgroup * 8) % 16
    row += (threadgroup // 4) * 4
    offsets = [(0, 0), (0, 1), (2, 0), (2, 1), (0, 4), (0, 5), (2, 4), (2, 5)]
    offset = offsets[register-16]
    row += offset[0]
    col += offset[1]
    thread_offsets = [(0, 0), (1, 0), (0, 2), (1, 2)]
    thread_offset = thread_offsets[thread % 4]
    row += thread_offset[0]
    col += thread_offset[1]
    if C_array[row, col] != 0:
        print("bad")
    return (row, col)


def check_sim_output():
    file = input("simulator output filename: ")

    A_array = np.zeros((16, 8))
    B_array = np.zeros((8, 16))
    C_array = np.zeros((16, 16))

    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            if "warp" in line:
                a, b, c = line.split(',')
                _, a = a.split(' ')
                _, b = b.strip().split(' ')
                c, d = c.strip().split(':')
                _, c = c.split(' ')
                warp = int(a)
                thread = int(b)
                register = int(c)
                value = d.strip()

                if warp != 0:
                    continue
                if not (32 <= register < 32+24):
                    continue

                register = register - 32

                # threadgroups 0, 4, 1, 5 have all elements of A
                threadgroup = thread // 4
                if threadgroup in [0, 4, 1, 5]:
                    row = [0, 4, 1, 5].index(threadgroup) * 4 + thread % 4
                    if 0 <= register < 8:
                        A_array[row, register] = hex2float(value)

                if threadgroup in [0, 4, 2, 6]:
                    col = [0, 4, 2, 6].index(threadgroup) * 4 + thread % 4
                    if 8 <= register < 16:
                        B_array[register-8, col] = hex2float(value)

                if 16 <= register < 24:
                    # print(value)
                    C_array[C_index(threadgroup, thread, register)] = hex2float(value)

    return [A_array, B_array, C_array]


if __name__ == "__main__":
    expected = np.load("abc.npz")
    # expected_A = expected['A_array']
    # expected_B = expected['B_array']
    # expected_C = expected['C_array']
    expected_A = expected['A_array'][0:8, 0:8]
    expected_B = expected['B_array'][0:8, 0:8]
    expected_C = expected['C_array'][0:8, 0:8]
    expected_C = expected_C + expected_A @ expected_B
    print('expected A:')
    print(expected_A)
    print('expected B:')
    print(expected_B)
    print('expected C:')
    print(expected_C[0:8, 0:8])
    expected_C.astype('float32').tofile("c_expected.bin")

    [got_A, got_B, got_C] = check_sim_output()
    print('got C:')
    print(C_array[0:8, 0:8])
    print('diff C:')
    print(expected_C[0:8, 0:8] - C_array[0:8, 0:8])
    assert np.allclose(expected_A, got_A)
    assert np.allclose(expected_B, got_B)
    assert np.allclose(expected_C, got_C)
