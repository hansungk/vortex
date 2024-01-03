__kernel void smemcoherence (__global volatile const int *src,
        __global volatile int *dst,
        __local volatile int *smem,
		int n)
{
    __local volatile int *markers = (__local int *)((__local unsigned char *)smem + 0x1000);
    int gid = get_global_id(0);

    // assumes total store ordering on smem
    markers[gid] = 0;
    smem[gid] = gid;
    markers[gid] = 1;

    // 0-th thread checks if all threads finished writing
    if (gid == 0) {
        int gridsize = get_global_size(0);
		int retry = 0;
		for (;; retry++) {
			for (int i = 0; i < gridsize; i++) {
				if (markers[i] != 1) {
					goto try_again;
				}
			}
			break;
		try_again:;
		}

		for (int i = 0; i < n; i++) {
			dst[i] = smem[i];
		}
		dst[n] = retry;
    }
}
