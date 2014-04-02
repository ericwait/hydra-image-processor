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
	bool setNthBuffCurent(int n);

private:
	CudaDeviceImages();
	CudaImageContainerClean** deviceImages;
	int getNextBuffNum();

	int numBuffers;
	int curBuff;
};