#pragma once
#include "CudaUtilities.h"

#include <vector>

class CudaDevices
{
public:
	CudaDevices() { reset(); }
	CudaDevices(int device) { reset(); getCudaInfo(device); }
	template <typename T>
	CudaDevices(T func, int device=-1)
	{
		reset();
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
	int getDeviceIdx(int deviceNum)
	{
		if (deviceNum >= deviceIdxList.size())
			return -1;

		return deviceIdxList.at(deviceNum);
	}
	
private:
	void reset();
	std::vector<int> deviceIdxList;
	size_t maxThreadsPerBlock;
	size_t availMem;
};
