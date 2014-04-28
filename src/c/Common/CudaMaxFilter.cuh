#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "CHelpers.h"
#include "CudaUtilities.cuh"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#else
//__constant__ extern float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
__global__ void cudaMaxFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, Vec<size_t> hostKernelDims,
							  PixelType minVal, PixelType maxVal)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double localMaxVal = imageIn[coordinate];
		DeviceVec<size_t> kernelDims = hostKernelDims;

		DeviceVec<int> startLimit = DeviceVec<int>(coordinate) - DeviceVec<int>((kernelDims)/2);
		DeviceVec<size_t> endLimit = coordinate + (kernelDims+1)/2;
		DeviceVec<size_t> kernelStart(DeviceVec<int>::max(-startLimit,DeviceVec<int>(0,0,0)));

		startLimit = DeviceVec<int>::max(startLimit,DeviceVec<int>(0,0,0));
		endLimit = DeviceVec<size_t>::min(DeviceVec<size_t>(endLimit),imageIn.getDeviceDims());

		DeviceVec<size_t> imageStart(coordinate-(kernelDims/2)+kernelStart);
		DeviceVec<size_t> iterationEnd(endLimit-DeviceVec<size_t>(startLimit));

		DeviceVec<size_t> i(0,0,0);
		for (i.z=0; i.z<iterationEnd.z; ++i.z)
		{
			for (i.y=0; i.y<iterationEnd.y; ++i.y)
			{
				for (i.x=0; i.x<iterationEnd.x; ++i.x)
				{
					if (cudaConstKernel[kernelDims.linearAddressAt(kernelStart+i)]==0)
						continue;

					double temp = imageIn[imageStart+i] * cudaConstKernel[kernelDims.linearAddressAt(kernelStart+i)];
					if (temp>localMaxVal)
					{
						localMaxVal = temp;
					}
				}
			}
		}

		imageOut[coordinate] = (localMaxVal>maxVal) ? (maxVal) : ((PixelType)localMaxVal);
	}
}

template <class PixelType>
PixelType* cMaxFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, PixelType** imageOut=NULL,
					 int device=0)
{
	PixelType* imOut = setUpOutIm(dims, imageOut);

	PixelType minVal = std::numeric_limits<PixelType>::lowest();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	if (kernel==NULL)
	{
		kernelDims = kernelDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernelDims.product()];
		for (int i=0; i<kernelDims.product(); ++i)
			ones[i] = 1.0f;

		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(float)*kernelDims.product()));
		delete[] ones;
	} 
	else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*kernelDims.product()));
	}

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,kernelDims);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMaxFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernelDims,
			minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}