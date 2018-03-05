#pragma once

#include "CudaImageContainer.cuh"
#include "KernelIterator.cuh"
#include "Vec.h"
#include <vector>
#include "CHelpers.h"
#include "CudaUtilities.cuh"
#include "ImageChunk.h"
#include "CudaDeviceImages.cuh"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
__global__ void cudaMaxFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, Vec<size_t> hostKernelDims, PixelType minVal, PixelType maxVal)
{
	Vec<size_t> coordinate_xyz;
	coordinate_xyz.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate_xyz.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate_xyz.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate_xyz<imageIn.getSpatialDims())
	{
		double localMaxVal = imageIn(coordinate_xyz);

		Vec<size_t> kernelDims = hostKernelDims;
		KernelIterator kIt(coordinate_xyz, imageIn.getDims(), kernelDims);

		for(; !kIt.end(); ++kIt)
		{
			float kernVal = cudaConstKernel[kIt.getKernelIdx()];

			if(kernVal==0)
				continue;

			Vec<float> imageCoord(kIt.getImageCoordinate());
			double temp = imageIn(imageCoord,kIt.getChannel(),kIt.getFrame()) * kernVal;

			if(temp>localMaxVal)
			{
				localMaxVal = temp;
			}
		}

		imageOut(coordinate_xyz) = (localMaxVal>maxVal) ? (maxVal) : ((PixelType)localMaxVal);
	}
}

template <class PixelType>
PixelType* cMaxFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, PixelType** imageOut=NULL, int device=-1)
{
    cudaSetDevice(device);
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

    int blockSize = getKernelMaxThreads(cudaMaxFilter<PixelType>);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,kernelDims,blockSize);

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