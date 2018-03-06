#include "CWrappers.h"
#include "ImageContainer.h"
#include "CudaDeviceCount.cuh"
#include "CudaDeviceStats.h"
#include "CudaMemoryStats.cuh"


void clearDevice()
{
	cudaDeviceReset();
}

 int deviceCount()
 {
	 return cDeviceCount();
 }

 int deviceStats(DevStats** stats)
 {
	 return cDeviceStats(stats);
 }

 int memoryStats(size_t** stats)
 {
	 return cMemoryStats(stats);
 }
