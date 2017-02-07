#pragma once

#include "Vec.h"
#include <limits>

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
__global__ void cudaLinearUnmixing(PixelType* imageIn, size_t imageDim, size_t numImages, PixelType minVal, PixelType maxVal)
{
	size_t pixelIdx = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ unsigned char sharedMemLin[];
	double* sharedM = (double*)sharedMemLin;

	if (pixelIdx < imageDim)
	{
		double* valueVector = sharedM + threadIdx.x*numImages;
		for (int chanOut=0; chanOut<numImages; ++chanOut)
		{
			valueVector[chanOut] = 0.0;
			for (int chanIn = 0; chanIn < numImages; ++chanIn)
			{
				valueVector[chanOut] += cudaConstKernel[chanOut + chanIn*numImages] * (double)(imageIn[pixelIdx + chanIn*imageDim]);
			}
		}

		for (int chanOut = 0; chanOut < numImages; ++chanOut)
		{
			PixelType valOut = (PixelType)fminf(fmaxf((float)(valueVector[chanOut]), minVal), maxVal);
			imageIn[pixelIdx + chanOut*imageDim] = valOut;
		}
	}
}

template <class PixelType>
PixelType* cLinearUnmixing(const PixelType* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, PixelType** imageOut, int device = 0)
{
	cudaSetDevice(device);
	PixelType* deviceIm;
	PixelType* imOut;
	if (imageOut == NULL)
		imOut = new PixelType[imageDims.product()*numImages];
	else
		imOut = *imageOut;

	memset(imOut, 0, imageDims.product()*numImages*sizeof(PixelType));

	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, unmixing, sizeof(float)*umixingDims.product()));

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);
	size_t memAvail, total;
	cudaMemGetInfo(&memAvail, &total);

	size_t numValsPerChunk = MIN(imageDims.product(), (size_t)((memAvail*MAX_MEM_AVAIL) / (numImages*sizeof(PixelType))));

	HANDLE_ERROR(cudaMalloc((void**)&deviceIm, sizeof(PixelType)*numValsPerChunk*numImages));
	PixelType min = std::numeric_limits<PixelType>::lowest();
	PixelType max = std::numeric_limits<PixelType>::max();

	for (size_t startIdx = 0; startIdx < imageDims.product(); startIdx+=numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk, imageDims.product() - startIdx);
		for (size_t chan = 0; chan < numImages; ++chan)
		{
			PixelType* deviceChanStart = deviceIm + curNumVals*chan;
			const PixelType* hostChanStart = imageIn + (imageDims.product()*chan + startIdx);
			HANDLE_ERROR(cudaMemcpy(deviceChanStart,hostChanStart,sizeof(PixelType)*curNumVals,cudaMemcpyHostToDevice));
		}
		
		int numBlocks = (int)(ceil((double)curNumVals / props.maxThreadsPerBlock));
		int maxThreads = (int)((double)props.sharedMemPerBlock / (sizeof(double)*numImages));
        int threads = getKernelMaxThreads(cudaLinearUnmixing<PixelType>,maxThreads);
		size_t sharedMemSize = sizeof(double)*threads*numImages;
		cudaLinearUnmixing<<<numBlocks,props.maxThreadsPerBlock,sharedMemSize>>>(deviceIm, curNumVals, numImages, min, max);
		DEBUG_KERNEL_CHECK();

		for (size_t chan = 0; chan < numImages; ++chan)
		{
			PixelType* deviceOutChan = deviceIm + curNumVals*chan;
			PixelType* imageOutChan = imOut + imageDims.product()*chan + startIdx;
			HANDLE_ERROR(cudaMemcpy(imageOutChan, deviceOutChan, sizeof(PixelType)*curNumVals, cudaMemcpyDeviceToHost));
		}
	}

	HANDLE_ERROR(cudaFree(deviceIm));

	return imOut;
}