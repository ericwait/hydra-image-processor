#pragma once
#include "cuda_runtime_api.h"

int cMemoryStats(std::size_t** stats)
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	*stats = new std::size_t[deviceCount*2];

	for(int i=0; i<deviceCount; ++i)
	{
		cudaSetDevice(i);

		std::size_t availMem,total;
		cudaMemGetInfo(&availMem,&total);
		(*stats)[i*2] = total;
		(*stats)[i*2+1] = availMem;
	}

	return deviceCount;
}
