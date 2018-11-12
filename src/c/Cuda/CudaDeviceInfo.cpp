#include "CudaDeviceInfo.h"
#include "CudaDeviceStats.h"
#include "Defines.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <signal.h>


extern "C" void HandleAborts(int signal_number)
{
	// Put a break point here to help with debugging
	fprintf(stderr, "Abort was called.");
}

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
			deviceIdxList[i] = i;
	}
	else
	{
		numDevices = 1;
		deviceIdxList.clear();
		deviceIdxList.push_back(device);
	}
	
	// Figure out the lowest memory available to make chunking less complicated
	// TODO: with some ambition, memory in ImageChunk could utilized each device independently
	for (int i=0; i<numDevices; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, deviceIdxList[i]);
		std::size_t mTPB = props.maxThreadsPerBlock;
		if (maxThreadsPerBlock > mTPB)
			maxThreadsPerBlock = mTPB;

		std::size_t curAvailMem = memoryAvailable(deviceIdxList[i]);
		if (availMem > curAvailMem)
			availMem = curAvailMem;

		std::size_t curSharedMem = props.sharedMemPerBlock;
		if (sharedMemPerBlock > curSharedMem)
			sharedMemPerBlock = curSharedMem;
	}

	availMem *= MAX_MEM_AVAIL;
}

void CudaDevices::reset()
{
	signal(SIGABRT, &HandleAborts);
	deviceIdxList.clear();
	availMem = ULLONG_MAX;
	maxThreadsPerBlock = ULLONG_MAX;
	sharedMemPerBlock = ULLONG_MAX;
}
