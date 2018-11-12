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
	void setMaxThreadsPerBlock(std::size_t newMax) { maxThreadsPerBlock = newMax; }

	std::size_t getMaxThreadsPerBlock()const { return maxThreadsPerBlock; }
	std::size_t getMinAvailMem()const { return availMem; }
	std::size_t getMinSharedMem()const { return sharedMemPerBlock; }
	std::size_t getNumDevices() const { return deviceIdxList.size(); }
	int getDeviceIdx(int deviceNum)
	{
		if (deviceNum >= deviceIdxList.size())
			deviceNum = deviceNum % deviceIdxList.size();

		int countDown = (int)deviceIdxList.size() - deviceNum - 1;
		return deviceIdxList.at(countDown);
	}
	
private:
	void reset();
	std::vector<int> deviceIdxList;
	std::size_t maxThreadsPerBlock;
	std::size_t availMem;
	std::size_t sharedMemPerBlock;
};
