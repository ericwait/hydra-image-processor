#pragma once
#include "CudaImageContainerClean.cuh"

template <typename PixelType>
class CudaDeviceImages
{
public:
	CudaDeviceImages(int numBuffers, Vec<size_t> maxDeviceDims, int device)
	{
		deviceImages = new CudaImageContainerClean<PixelType>*[numBuffers];

		for (int i=0; i<numBuffers; ++i)
			deviceImages[i] = new CudaImageContainerClean<PixelType>(maxDeviceDims,device);

		this->numBuffers = numBuffers;
		curBuff = 0;
	}

	~CudaDeviceImages()
	{
		for (int i=0; i<numBuffers; ++i)
			delete deviceImages[i];
	}

	CudaImageContainer<PixelType>* getCurBuffer()
	{
		if (curBuff<numBuffers)
			return deviceImages[curBuff];

		return NULL;
	}

	CudaImageContainer<PixelType>* getNextBuffer()
	{
		if (numBuffers<2)
			return NULL;

		return deviceImages[getNextBuffNum()];
	}

	CudaImageContainer<PixelType>* getThirdBuffer()
	{
		if (numBuffers<3)
			return NULL;

		int trd = getNextBuffNum() + 1;

		if (trd >= numBuffers)
			trd = 0;

		return deviceImages[trd];
	}

	void incrementBuffer()
	{
		++curBuff;
		if (curBuff >= numBuffers)
			curBuff = 0;
	}

	void setAllDims(Vec<size_t> dims)
	{
		for (int i=0; i<numBuffers; ++i)
			deviceImages[i]->setDims(dims);
	}

	void setNextDims(Vec<size_t> dims)
	{
		if (numBuffers>1)
			deviceImages[getNextBuffNum()]->setDims(dims);
	}

	bool setNthBuffCurent(int n)
	{
		if (n>numBuffers)
			return false;

		int nth = curBuff;

		for (int i=1; i<n; ++i)
		{
			++nth;
			if (nth >= numBuffers)
				nth = 0;
		}

		curBuff = nth;

		return true;
	}

private:
	CudaDeviceImages();
	CudaImageContainerClean<PixelType>** deviceImages;
	int getNextBuffNum()
	{
		int next = curBuff + 1;

		if (next>=numBuffers)
			next = 0;

		return next;
	}

	int numBuffers;
	int curBuff;
};