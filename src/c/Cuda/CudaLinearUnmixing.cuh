#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "Vec.h"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
__global__ void cudaLinearUnmixing(PixelType* imageIn, size_t imageDim, size_t numImages, float* imagesOut)
{
	size_t pixelIdx = threadIdx.x + blockIdx.x*blockDim.x;

	if (pixelIdx < imageDim)
	{
		for (int chanOut=0; chanOut<numImages; ++chanOut)
		{
			//imagesOut[pixelIdx] = imageIn[pixelIdx];
			double val = 0.0;
			for (int chanIn = 0; chanIn < numImages; ++chanIn)
			{
				val += cudaConstKernel[chanOut + chanIn*numImages] * (double)(imageIn[pixelIdx + chanIn*imageDim]);
			}

			imagesOut[pixelIdx + chanOut*imageDim] = (float)val;
		}
	}
}

template <class PixelType>
float* cLinearUnmixing(const PixelType* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, float** imageOut, int device = 0)
{
	PixelType* deviceIn;
	float* imOut, * deviceOut;
	if (imageOut == NULL)
		imOut = new float[imageDims.product()*numImages];
	else
		imOut = *imageOut;

	memset(imOut, 0, imageDims.product()*numImages*sizeof(float));

	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, unmixing, sizeof(float)*umixingDims.product()));

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);
	size_t memAvail, total;
	cudaMemGetInfo(&memAvail, &total);

	size_t numValsPerChunk = MIN(imageDims.product(), (size_t)((memAvail*MAX_MEM_AVAIL) / (numImages*(sizeof(PixelType) + sizeof(float)))));

	HANDLE_ERROR(cudaMalloc((void**)&deviceIn, sizeof(PixelType)*numValsPerChunk*numImages));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOut, sizeof(float)*numValsPerChunk*numImages));

	for (size_t startIdx = 0; startIdx < imageDims.product(); startIdx+=numValsPerChunk)
	{
 		size_t curNumVals = MIN(numValsPerChunk, imageDims.product() - startIdx);
		for (size_t chan = 0; chan < numImages; ++chan)
		{
			PixelType* deviceChanStart = deviceIn + numValsPerChunk*chan;
			const PixelType* hostChanStart = imageIn + (imageDims.product()*chan + startIdx);
			HANDLE_ERROR(cudaMemcpy(deviceChanStart,hostChanStart,sizeof(PixelType)*curNumVals,cudaMemcpyHostToDevice));
		}
		
		int numBlocks = (int)(ceil((double)curNumVals / props.maxThreadsPerBlock));
		cudaLinearUnmixing<<<numBlocks,props.maxThreadsPerBlock>>>(deviceIn, curNumVals, numImages, deviceOut);
		DEBUG_KERNEL_CHECK();

		for (size_t chan = 0; chan < numImages; ++chan)
		{
			float* deviceOutChan = deviceOut + curNumVals*chan;
			float* imageOutChan = imOut + imageDims.product()*chan + startIdx;
			HANDLE_ERROR(cudaMemcpy(imageOutChan, deviceOutChan, sizeof(float)*curNumVals, cudaMemcpyDeviceToHost));
		}
	}

	HANDLE_ERROR(cudaFree(deviceOut));
	HANDLE_ERROR(cudaFree(deviceIn));

	return imOut;
}