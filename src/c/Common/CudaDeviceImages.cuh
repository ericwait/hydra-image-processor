#pragma once
#include "CudaImageContainerClean.cuh"

class CudaDeviceImages
{
public:
	CudaDeviceImages(int numBuffers, Vec<size_t> maxDeviceDims, int device);
	~CudaDeviceImages();

	CudaImageContainer<DevicePixelType>* getCurBuffer();
	CudaImageContainer<DevicePixelType>* getNextBuffer();
	CudaImageContainer<DevicePixelType>* getThirdBuffer();
	void incrementBuffer();
	void setAllDims(Vec<size_t> dims);
	void setNextDims(Vec<size_t> dims);
	bool setNthBuffCurent(int n);

private:
	CudaDeviceImages();
	CudaImageContainerClean<DevicePixelType>** deviceImages;
	int getNextBuffNum();

	int numBuffers;
	int curBuff;
};