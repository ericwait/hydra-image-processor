#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "CHelpers.h"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"

template <class PixelType>
__global__ void cudaMultiplyImageScaler( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, double factor,
								  PixelType minValue, PixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double outValue = factor*imageIn[coordinate];
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}

template <class PixelType>
__global__ void cudaMultiplyTwoImages(CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageIn2,
									  CudaImageContainer<PixelType> imageOut, double factor, PixelType minValue,
									  PixelType maxValue)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double outValue = factor * (double)(imageIn1[coordinate]) * (double)(imageIn2[coordinate]);
		imageOut[coordinate] = (outValue>(double)maxValue) ? (maxValue) : ((outValue<(double)minValue) ? (minValue) : (outValue));
	}
}

template <class PixelType>
PixelType* multiplyImage(const PixelType* imageIn, Vec<size_t> dims, double multiplier, PixelType** imageOut=NULL, int device=0)
{
	PixelType* imOut = setUpOutIm(dims, imageOut);

	PixelType minVal = std::numeric_limits<PixelType>::lowest();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMultiplyImageScaler<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			multiplier,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}