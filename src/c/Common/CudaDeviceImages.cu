#include "CudaDeviceImages.cuh"


CudaDeviceImages::CudaDeviceImages(int numBuffers, Vec<size_t> maxDeviceDims, int device)
{
	deviceImages = new CudaImageContainerClean*[numBuffers];

	for (int i=0; i<numBuffers; ++i)
		deviceImages[i] = new CudaImageContainerClean(maxDeviceDims,device);

	this->numBuffers = numBuffers;
	curBuff = 0;
	nextBuff = 1;
}

CudaDeviceImages::~CudaDeviceImages()
{
	for (int i=0; i<numBuffers; ++i)
		delete deviceImages[i];
}

CudaImageContainer* CudaDeviceImages::getCurBuffer()
{
	if (curBuff<numBuffers)
		return deviceImages[curBuff];

	return NULL;
}

CudaImageContainer* CudaDeviceImages::getNextBuffer()
{
	if (nextBuff<numBuffers)
		return deviceImages[nextBuff];
	
	return NULL;
}

CudaImageContainer* CudaDeviceImages::getThirdBuffer()
{
	if (numBuffers<3)
		return NULL;

	int trd = nextBuff + 1;

	if (trd >= numBuffers)
		trd = 0;

	return deviceImages[trd];
}

void CudaDeviceImages::setAllDims(Vec<size_t> dims)
{
	for (int i=0; i<numBuffers; ++i)
		deviceImages[i]->setDims(dims);
}


void CudaDeviceImages::setNextDims(Vec<size_t> dims)
{
	if (nextBuff<numBuffers)
		deviceImages[nextBuff]->setDims(dims);
}

void CudaDeviceImages::incrementBuffer()
{
	if (++curBuff >= numBuffers)
		curBuff = 0;

	if (++nextBuff >= numBuffers)
		nextBuff = 0;
}

bool CudaDeviceImages::setNthBuffCurent(int n)
{
	if (n>numBuffers)
		return false;

	int nth = curBuff;

	for (int i=1; i<n; ++i)
	{
		++nth;
		if (nth > numBuffers)
			nth = 0;
	}

	curBuff = nth;

	return true;
}
