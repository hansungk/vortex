__kernel void vecadd_loop (__global volatile const float *A,
	                    __global volatile const float *B,
	                    __global volatile float *C)
{
  int gid = get_global_id(0);
  float sum = 0.;
  for (int i = 0; i < 500; i++) {
      // int addr = gid + (i % 2);
	  int addr = gid;
      C[addr] += A[addr] + B[addr];
  }
  // C[gid] = sum;
}
