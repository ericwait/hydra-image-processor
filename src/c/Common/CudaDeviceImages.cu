#include "CudaDeviceImages.cuh"


CudaDeviceImages::CudaDeviceImages(int numBuffers, Vec<size_t> maxDeviceDims, int device)
{
	deviceImages = new CudaImageContainerClean*[numBuffers];

	for (int i=0; i<numBuffers; ++i)
		deviceImages[i] = new CudaImageContainerClean(maxDeviceDims,device);

	this->numBuffers = numBuffers;
	curBuff = 0;
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
	if (numBuffers<2)
		return NULL;

	return deviceImages[getNextBuffNum()];
}

CudaImageContainer* CudaDeviceImages::getThirdBuffer()
{
	if (numBuffers<3)
		return NULL;

	int trd = getNextBuffNum() + 1;

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
	if (numBuffers>1)
		deviceImages[getNextBuffNum()]->setDims(dims);
}

void CudaDeviceImages::incrementBuffer()
{
	++curBuff;
	if (curBuff >= numBuffers)
		curBuff = 0;
}

bool CudaDeviceImages::setNthBuffCurent(int n)
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

int CudaDeviceImages::getNextBuffNum()
{
	int next = curBuff + 1;

	if (next>=numBuffers)
		next = 0;

	return next;
}
