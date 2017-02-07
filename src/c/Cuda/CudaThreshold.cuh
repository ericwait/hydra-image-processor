#pragma once

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "CHelpers.h"
#include "CudaHistogram.cuh"

template <class PixelType>
__global__ void cudaThreshold( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, PixelType threshold,
							  PixelType minValue, PixelType maxValue )
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		imageOut[coordinate] = (imageIn[coordinate]>=threshold) ? (maxValue) : (minValue);
	}
}

template <class PixelType>
PixelType* cThresholdFilter(const PixelType* imageIn, Vec<size_t> dims, PixelType thresh, PixelType** imageOut=NULL, int device=0)
{
    cudaSetDevice(device);
	PixelType* imOut = setUpOutIm(dims, imageOut);

	PixelType minVal = std::numeric_limits<PixelType>::min();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

    int blockSize = getKernelMaxThreads(cudaThreshold<PixelType>);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,Vec<size_t>(0,0,0),blockSize);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaThreshold<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			thresh,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

template <class PixelType>
PixelType* cOtsuThresholdFilter(const PixelType* imageIn, Vec<size_t> dims, double alpha=1.0, PixelType** imageOut=NULL, int device=0)
{
    cudaSetDevice(device);
	PixelType thresh = cOtsuThresholdValue(imageIn,dims,device);
	thresh = (PixelType)(thresh*alpha);

	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}