#pragma once
#include "CudaImageContainerClean.cuh"

class CudaDeviceImages
{
public:
	CudaDeviceImages(int numBuffers, Vec<size_t> maxDeviceDims, int device);
	~CudaDeviceImages();

	CudaImageContainer* getCurBuffer();
	CudaImageContainer* getNextBuffer();
	CudaImageContainer* getThirdBuffer();
	void incrementBuffer();
	void setAllDims(Vec<size_t> dims);
	void setNextDims(Vec<size_t> dims);

private:
	CudaDeviceImages();
	CudaImageContainerClean** deviceImages;

	int numBuffers;
	int curBuff;
	int nextBuff;
};