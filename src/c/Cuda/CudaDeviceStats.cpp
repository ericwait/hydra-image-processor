#include "CudaDeviceStats.h"
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