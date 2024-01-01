__kernel void sharedmem (__global volatile const float *src,
						 __global volatile float *dst,
						 __local volatile float *smem)
{
  int gid = get_global_id(0);
  smem[gid] = src[gid];
  float read;
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 5000; i++) {
	  read = smem[gid];
  }
  dst[gid] = read;
}
