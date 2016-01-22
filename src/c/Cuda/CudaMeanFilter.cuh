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
__global__ void cudaMeanFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, Vec<size_t> hostKernelDims )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double val = 0;
		double kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> halfKernal = kernelDims/2;

		DeviceVec<size_t> curCoordIm = coordinate - halfKernal;
		curCoordIm.z = (coordinate.z<halfKernal.z) ? 0 : coordinate.z-halfKernal.z;
		for (; curCoordIm.z<=coordinate.z+halfKernal.z && curCoordIm.z<imageIn.getDeviceDims().z; ++curCoordIm.z)
		{
			curCoordIm.y = (coordinate.y<halfKernal.y) ? 0 : coordinate.y-halfKernal.y;
			for (; curCoordIm.y<=coordinate.y+halfKernal.y && curCoordIm.y<imageIn.getDeviceDims().y; ++curCoordIm.y)
			{
				curCoordIm.x = (coordinate.x<halfKernal.x) ? 0 : coordinate.x-halfKernal.x;
				for (; curCoordIm.x<=coordinate.x+halfKernal.x && curCoordIm.x<imageIn.getDeviceDims().x; ++curCoordIm.x)
				{
					val += imageIn[curCoordIm];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinate] = val/kernelVolume;
	}
}

template <class PixelType>
PixelType* cMeanFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, PixelType** imageOut=NULL, int device=0)
{
	PixelType* imOut = setUpOutIm(dims, imageOut);

	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t memAvail, total;
	cudaMemGetInfo(&memAvail,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(memAvail*MAX_MEM_AVAIL),props,neighborhood);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMeanFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),neighborhood);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}