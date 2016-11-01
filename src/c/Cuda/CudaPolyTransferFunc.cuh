#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "ImageChunk.cuh"
#include "CHelpers.h"
#include "CudaDeviceImages.cuh"

template <class PixelType>
__global__ void cudaPolyTransferFunc( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, double a, double b,
									 double c, PixelType minPixelValue, PixelType maxPixelValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double pixVal = (double)imageIn[coordinate] / maxPixelValue;// place value between [0,1]
		double multiplier = a*pixVal*pixVal + b*pixVal + c;
		if (multiplier<0)
			multiplier = 0;
		if (multiplier>1)
			multiplier = 1;

		PixelType newPixelVal = min((double)maxPixelValue,max((double)minPixelValue, multiplier*maxPixelValue));

		imageOut[coordinate] = newPixelVal;
	}
}

template <class PixelType>
PixelType* cApplyPolyTransferFunction(const PixelType* imageIn, Vec<size_t> dims, double a, double b, double c,
									 PixelType minValue=std::numeric_limits<PixelType>::lowest(),
									 PixelType maxValue=std::numeric_limits<PixelType>::max(), PixelType** imageOut=NULL, int device=0)
{
    cudaSetDevice(device);
	PixelType* imOut = setUpOutIm(dims, imageOut);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t memAvail, total;
	cudaMemGetInfo(&memAvail,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(memAvail*MAX_MEM_AVAIL),props);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaPolyTransferFunc<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			a,b,c,minValue,maxValue);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}
