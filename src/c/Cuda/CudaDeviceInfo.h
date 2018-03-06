#pragma once
#include "CudaUtilities.h"

#include <vector>

class CudaDevices
{
public:
	CudaDevices() { reset(); }
	CudaDevices(int device) { getCudaInfo(device); }
	template <typename T>
	CudaDevices(T func, int device=-1)
	{
		getCudaInfo(device);
		setThreadsForFunction(func);
	}

	~CudaDevices() { reset(); }

	void getCudaInfo(int device = -1);

	template <typename T>
	void setThreadsForFunction(T func)
	{
		maxThreadsPerBlock = MIN(maxThreadsPerBlock, getKernelMaxThreads(func));
	}

	size_t getMaxThreadsPerBlock()const { return maxThreadsPerBlock; }
	size_t getMinAvailMem()const { return availMem; }
	size_t getNumDevices() const { return deviceIdxList.size(); }
	
private:
	void reset();
	std::vector<int> deviceIdxList;
	size_t maxThreadsPerBlock;
	size_t availMem;
};
