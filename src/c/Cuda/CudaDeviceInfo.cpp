#include "CudaDeviceInfo.h"
#include "CudaDeviceStats.h"
#include "Defines.h"

#include <cuda_runtime.h>
#include <limits.h>


void CudaDevices::getCudaInfo(int device/*=-1*/)
{
	// Get device count
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	if (numDevices==0)
	{
		reset();
		return;
	}
	
	// Make a list of devices available
	// If the device number was explicitly passed, just use that one
	if (device < 0)
	{
		deviceIdxList.resize(numDevices);
		for (unsigned char i = 0; i < numDevices; ++i)
			deviceIdxList.push_back(i);
	}
	else
	{
		numDevices = 1;
		deviceIdxList.resize(numDevices);
		deviceIdxList.push_back(device);
	}
	
	// Figure out the lowest memory available to make chunking less complicated
	// TODO: with some ambition, memory in ImageChunk could utilized each device independently
	availMem = ULLONG_MAX;
	for (int i=0; i<numDevices; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, deviceIdxList[i]);
		size_t mTPB = props.maxThreadsPerBlock;
		if (maxThreadsPerBlock > mTPB)
			maxThreadsPerBlock = mTPB;

		size_t availMem = memoryAvailable(deviceIdxList[i]);
		if (availMem > availMem)
			availMem = availMem;
	}

	availMem *= MAX_MEM_AVAIL;
}

void CudaDevices::reset()
{
	deviceIdxList.clear();
	availMem = 0;
	maxThreadsPerBlock = 0;
}
