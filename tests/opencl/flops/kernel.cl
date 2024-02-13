__kernel void flops (__global volatile const float *src,
						 __global volatile float *dst,
						 __local volatile float *smem)
{
  int gid = get_global_id(0);
  float f = 0.0f;
  float incr = src[0];
  __attribute__((opencl_unroll_hint(16)))
  for (int i = 0; i < 5000; i++) {
	  f += incr;
  }
  dst[gid] = f;
}
