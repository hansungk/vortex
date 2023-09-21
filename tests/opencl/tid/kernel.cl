__kernel void tid()
{
  __global int *out = (__global int *)0xc0000000;
  int gid = get_global_id(0);
  out[gid] = gid;
}