#pragma once
#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "CHelpers.h"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"

template <class PixelType>
__global__ void cudaClamp(CudaImageContainer<PixelType> imageIn,CudaImageContainer<PixelType> imageOut,	PixelType minValue,PixelType maxValue)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if(coordinate<imageIn.getDeviceDims())
	{
		imageOut[coordinate] = (imageIn[coordinate]>maxValue) ? (maxValue) : ((imageIn[coordinate]<minValue) ? (minValue) : (imageIn[coordinate]));
	}
}

template <class PixelType>
PixelType* cClamp(const PixelType* imageIn,Vec<size_t> dims,PixelType minVal,PixelType maxVal,PixelType** imageOut=NULL,int device=0)
{
    cudaSetDevice(device);

	PixelType* imOut = setUpOutIm(dims,imageOut);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem,total;
	cudaMemGetInfo(&availMem,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks,maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for(std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());

		cudaClamp<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		curChunk->retriveROI(imOut,dims,deviceImages.getNextBuffer());
	}

	return imOut;
}