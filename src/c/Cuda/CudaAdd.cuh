#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "CHelpers.h"
#include "ImageChunk.h"
#include "CudaDeviceImages.cuh"
#include "ImageContainer.h"
#include "CudaUtilities.cuh"

template <class PixelType>
__global__ void cudaAddScaler( CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageOut, double factor, PixelType minValue, PixelType maxValue )
{
	Vec<size_t> coordinate = GetThreadBlockCoordinate();

	if (coordinate<imageIn1.getDims())
	{
		KernelIterator kIt(coordinate, imageIn1.getDims(), Vec<size_t>(1));
		for(; !kIt.end(); ++kIt)
		{
			double outValue = imageIn1(kIt.getFullPos())+factor;
			imageOut(kIt.getFullPos()) = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
		}
	}
}

template <class PixelType>
__global__ void cudaAddTwoImagesWithFactor( CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageIn2, CudaImageContainer<PixelType> imageOut, double factor, PixelType minValue, PixelType maxValue )
{
	Vec<size_t> coordinate = GetThreadBlockCoordinate();

	if (coordinate<imageIn1.getDims())
	{
		KernelIterator kIt(coordinate, imageIn1.getDims(), Vec<size_t>(1));
		for(; !kIt.end(); ++kIt)
		{
			double additive = factor* double(imageIn2(kIt.getFullPos()));
			double outValue = double(imageIn1(kIt.getFullPos())) + additive;

			imageOut(kIt.getFullPos()) = (outValue>(double)maxValue) ? (maxValue) : ((outValue<(double)minValue) ? (minValue) : ((PixelType)outValue));
		}
	}
}

template <class PixelTypeIn, class PixelTypeOut>
void cAddConstant(const ImageContainer<PixelTypeIn> imageIn, double additive, ImageContainer<PixelTypeOut>& imageOut, int device=-1)
{
	const int NUM_BUFF_NEEDED = 2;
	setUpOutIm(imageIn.getDims(), imageOut);

	PixelTypeOut minVal = std::numeric_limits<PixelTypeOut>::lowest();
	PixelTypeOut maxVal = std::numeric_limits<PixelTypeOut>::max();

	int* deviceIdxList;
	int numDevices;
	size_t maxThreadsPerBlock;
	size_t availMem = getCudaInfo(&deviceIdxList, numDevices, maxThreadsPerBlock, device);

    int blockSize = getKernelMaxThreads(cudaAddScaler<PixelTypeOut>);
	maxThreadsPerBlock = MIN(maxThreadsPerBlock, size_t(blockSize));

    std::vector<ImageChunk> chunks = calculateBuffers<PixelTypeOut>(imageIn.getDims(), NUM_BUFF_NEEDED, (size_t)(availMem*MAX_MEM_AVAIL), sizeof(PixelTypeOut), maxThreadsPerBlock, Vec<size_t>(0));

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	{// This is were the openMP should go
		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, deviceIdxList[0]);

		for(std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk!=chunks.end(); ++curChunk)
		{
			curChunk->sendROI(imageIn, deviceImages.getCurBuffer());
			deviceImages.setNextDims(curChunk->getFullChunkSize());

			cudaAddScaler<<<curChunk->blocks, curChunk->threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), additive, minVal, maxVal);
			DEBUG_KERNEL_CHECK();

			deviceImages.incrementBuffer();
			curChunk->retriveROI(imOut, deviceImages.getCurBuffer());
		}
	}
}

/*
*	Adds this image to the passed in one.  You can apply a factor
*	which is multiplied to the passed in image prior to adding
*/
template <class PixelType>
void cAddImageWith(const ImageContainer<PixelType> imageIn1, const ImageContainer<PixelType> imageIn2, double additive, ImageContainer<PixelType> imageOut, int device=-1)
{

	const int NUM_BUFF_NEEDED = 3;
	setUpOutIm(imageIn.getDims(), imageOut);

	PixelType minVal = std::numeric_limits<PixelType>::lowest();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	int* deviceIdxList;
	int numDevices;
	size_t maxThreadsPerBlock;
	size_t availMem = getCudaInfo(&deviceIdxList, numDevices, maxThreadsPerBlock, device);

	int blockSize = getKernelMaxThreads(cudaAddTwoImagesWithFactor<PixelType>);
	maxThreadsPerBlock = MIN(maxThreadsPerBlock, size_t(blockSize));

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(imageIn.getDims(), NUM_BUFF_NEEDED, (size_t)(availMem*MAX_MEM_AVAIL), sizeof(PixelType), maxThreadsPerBlock, Vec<size_t>(0));

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	{// This is were the openMP should go
		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, deviceIdxList[0]);

		for(std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk!=chunks.end(); ++curChunk)
		{
			deviceImages.setAllDims(curChunk->getFullChunkSize());
			curChunk->sendROI(imageIn1, deviceImages.getCurBuffer());
			curChunk->sendROI(imageIn1, deviceImages.getNextBuffer());

			cudaAddTwoImagesWithFactor<<<curChunk->blocks, curChunk->threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()),*(deviceImages.getThirdBuffer()), additive, minVal, maxVal);
			DEBUG_KERNEL_CHECK();

			deviceImages.incrementBuffer();
			curChunk->retriveROI(imOut, deviceImages.getThirdBuffer());
		}
	}
}
