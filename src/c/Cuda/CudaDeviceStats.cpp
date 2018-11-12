#include "CudaDeviceStats.h"
#include "CudaUtilities.h"

#include <cuda_runtime.h>

int cDeviceStats(DevStats** stats)
{
    int cnt = 0;
    cudaGetDeviceCount(&cnt);

    *stats = new DevStats[cnt];

    for(int device = 0; device<cnt; ++device)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        DevStats* curStats = (*stats)+device;
        curStats->name = props.name;
        curStats->major = props.major;
        curStats->minor = props.minor;

        curStats->constMem = props.totalConstMem;
        curStats->sharedMem = props.sharedMemPerBlock;
        curStats->totalMem = props.totalGlobalMem;

        curStats->tccDriver = props.tccDriver>0;
        curStats->mpCount = props.multiProcessorCount;
        curStats->threadsPerMP = props.maxThreadsPerMultiProcessor;

        curStats->warpSize = props.warpSize;
        curStats->maxThreads = props.maxThreadsPerBlock;
    }

    return cnt;
}

std::size_t memoryAvailable(int device, std::size_t* totalOut/*=NULL*/)
{
	HANDLE_ERROR(cudaSetDevice(device));
	std::size_t free, total;
	HANDLE_ERROR(cudaMemGetInfo(&free, &total));

	if (totalOut != NULL)
		*totalOut = total;

	return free;
}

bool checkFreeMemory(std::size_t needed, int device, bool throws/*=false*/)
{
	std::size_t free = memoryAvailable(device);
	if (needed > free)
	{
		if (throws)
		{
			char buff[255];
			sprintf(buff, "Out of CUDA Memory!\nNeed: %zu\nHave: %zu\n", needed, free);
			throw std::runtime_error(buff);
		}
		return false;
	}
	return true;
}
