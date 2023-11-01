__kernel void vecadd (__global const float *A,
	                    __global const float *B,
	                    __global float *C)
{
  int gid = get_global_id(0);
  float sum = 0.;
  for (int i = 0; i < 1000; i++) {
	  int addr = gid + (i % 2);
	  sum += A[addr] + B[addr];
  }
  C[gid] = sum;
}
