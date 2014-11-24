#pragma once
#include "cuda_runtime_api.h"

int cMemoryStats(size_t** stats)
{
	int cnt = 0;
	cudaGetDeviceCount(&cnt);

	*stats = new size_t[cnt*2];

	for(int i=0; i<cnt; ++i)
	{
		cudaSetDevice(i);

		size_t availMem,total;
		cudaMemGetInfo(&availMem,&total);
		(*stats)[i*2] = total;
		(*stats)[i*2+1] = availMem;
	}

	return cnt;
}