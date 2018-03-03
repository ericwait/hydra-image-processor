#include "CudaUtilities.h"
#include <limits.h>


size_t getCudaInfo(int** deviceIdxList, int& numDevices, size_t& maxThreadsPerBlock, int device/*=-1*/)
{
	// Get device count
	cudaGetDeviceCount(&numDevices);
	
	// Make a list of devices available
	// If the device number was explicitly passed, just use that one
	int* _deviceIdxList; 
	if (device < 0)
	{
		_deviceIdxList = new int[numDevices];
		for (unsigned char i = 0; i < numDevices; ++i)
			_deviceIdxList[i] = i;
	}
	else
	{
		numDevices = 1;
		_deviceIdxList = new int[1];
		_deviceIdxList[0] = device;
	}
	
	// Figure out the lowest memory available to make chunking less complicated
	// TODO: with some ambition, memory in the chunking could utilized each device independently
	size_t minAvailMem = ULLONG_MAX;
	size_t minThreadsPerBlock = ULLONG_MAX;
	for (int i=0; i<numDevices; ++i)
	{
		cudaDeviceProp* props;
		cudaGetDeviceProperties(&props, _deviceIdxList[i]);
		size_t mTPB = props->maxThreadsPerBlock;
		if (minThreadsPerBlock > mTPB)
			minThreadsPerBlock = mTPB;

		size_t availMem = memoryAvailable(_deviceIdxList[i]);
		if (minAvailMem > availMem)
			minAvailMem = availMem;
	}

	// Set return values
	*deviceIdxList = _deviceIdxList;
	maxThreadsPerBlock = minThreadsPerBlock;

	return minAvailMem;
}
