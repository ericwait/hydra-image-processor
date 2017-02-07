#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "CHelpers.h"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"

template <class PixelType>
__global__ void cudaAddScaler( CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageOut, double factor, 
							  PixelType minValue, PixelType maxValue )
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		double outValue = imageIn1[coordinate] + factor;
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}

template <class PixelType>
__global__ void cudaAddTwoImagesWithFactor( CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageIn2,
										   CudaImageContainer<PixelType> imageOut, double factor, PixelType minValue, PixelType maxValue )
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		double additive = factor*(double)(imageIn2[coordinate]);
		double outValue = (double)(imageIn1[coordinate]) + additive;

		imageOut[coordinate] = (outValue>(double)maxValue) ? (maxValue) : ((outValue<(double)minValue) ? (minValue) : ((PixelType)outValue));
	}
}

template <class PixelTypeIn, class PixelTypeOut>
PixelTypeOut* cAddConstant(const PixelTypeIn* imageIn, Vec<size_t> dims, double additive, PixelTypeOut** imageOut=NULL, int device=0)
{
	PixelTypeOut* imOut = setUpOutIm(dims, imageOut);

	PixelTypeOut minVal = std::numeric_limits<PixelTypeOut>::lowest();
	PixelTypeOut maxVal = std::numeric_limits<PixelTypeOut>::max();

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

    int blockSize = getKernelMaxThreads(cudaAddScaler<PixelTypeOut>);

    std::vector<ImageChunk> chunks = calculateBuffers<PixelTypeOut>(dims, 2, (size_t)(availMem*MAX_MEM_AVAIL), props,Vec<size_t>(0, 0, 0), blockSize);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelTypeOut> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaAddScaler<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			additive,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

/*
*	Adds this image to the passed in one.  You can apply a factor
*	which is multiplied to the passed in image prior to adding
*/
template <class PixelType>
PixelType* cAddImageWith(const PixelType* imageIn1, const PixelType* imageIn2, Vec<size_t> dims, double additive,
						PixelType** imageOut=NULL, int device=0)
{
    cudaSetDevice(device);

	PixelType* imOut = setUpOutIm(dims, imageOut);

	PixelType minVal = std::numeric_limits<PixelType>::lowest();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

    int blockSize = getKernelMaxThreads(cudaAddTwoImagesWithFactor<PixelType>);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,3,(size_t)(availMem*MAX_MEM_AVAIL),props,Vec<size_t>(0,0,0),blockSize);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(3,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());
		curChunk->sendROI(imageIn1,dims,deviceImages.getCurBuffer());
		curChunk->sendROI(imageIn2,dims,deviceImages.getNextBuffer());

		cudaAddTwoImagesWithFactor<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			*(deviceImages.getThirdBuffer()),additive,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		curChunk->retriveROI(imOut,dims,deviceImages.getThirdBuffer());
	}

	return imOut;
}