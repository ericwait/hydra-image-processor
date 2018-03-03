#pragma once

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
#endif

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaNeighborhoodSum( CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, Vec<size_t> hostKernelDims,
							  PixelTypeOut minVal, PixelTypeOut maxVal)
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		double sumVal = imageIn(coordinate);
		Vec<size_t> kernelDims = hostKernelDims;
		KernelIterator kIt(coordinate, imageIn.getDims(), kernelDims);

		for (; !kIt.end(); ++kIt)
		{
			Vec<size_t> kernIdx(kIt.getKernelIdx());
			float kernVal = cudaConstKernel[kernelDims.linearAddressAt(kernIdx)];
			if (kernVal == 0)
				continue;

			sumVal += imageIn(kIt.getImageCoordinate()) * kernVal;
		}

		imageOut(coordinate) = (sumVal>maxVal) ? (maxVal) : ((PixelTypeOut)sumVal);
	}
}

template <class PixelTypeIn,class PixelTypeOut>
PixelTypeOut* cNeighborhoodSum(const PixelTypeIn* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, PixelTypeOut** imageOut=NULL,
					 int device=0)
{
    cudaSetDevice(device);
	PixelTypeOut* imOut = setUpOutIm(dims, imageOut);

	PixelTypeOut minVal = std::numeric_limits<PixelTypeOut>::lowest();
	PixelTypeOut maxVal = std::numeric_limits<PixelTypeOut>::max();

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

    int blockSize = getKernelMaxThreads(cudaMaxFilter<PixelTypeOut>);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelTypeOut>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,kernelDims,blockSize);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelTypeOut> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaNeighborhoodSum <<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernelDims,
			minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}