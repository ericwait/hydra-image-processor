#pragma once
#include "CudaImageContainer.cuh"

#include "Vec.h"
#include <vector>
#include "CudaUtilities.cuh"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "Defines.h"

template <class PixelType>
__global__ void cudaHistogramCreate( PixelType* values, size_t numValues, size_t* histogram, PixelType minVal, double binSize,
									unsigned int numBins)
{
	//This code is modified from that of Sanders - Cuda by Example
	extern __shared__ size_t tempHisto[];

	if (threadIdx.x<numBins)
		tempHisto[threadIdx.x] = 0;

	__syncthreads();

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < numValues)
	{
		size_t binNum = (size_t)MAX( 0.0, ( (double)(values[i])-minVal) / binSize );
		binNum = MIN(binNum, (size_t)numBins-1);
		atomicAdd(&(tempHisto[binNum]), 1);
		i += stride;
	}

	__syncthreads();
	if (threadIdx.x<numBins)
		atomicAdd(&(histogram[threadIdx.x]), tempHisto[threadIdx.x]);
}

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, unsigned int numBins, double devisor)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i<numBins)
	{
		normHistogram[i] = histogram[i] / devisor;
		i += stride;
	}
}

template <class PixelType>
size_t* createHistogram(int device, unsigned int arraySize, Vec<size_t> dims, PixelType maxVal, PixelType minVal, const PixelType* imageIn)
{
	cudaSetDevice(device);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	if ((size_t)props.sharedMemPerBlock<sizeof(size_t)*arraySize)
		throw std::runtime_error("Too many bins to calculate on GPU with current shared memory constraints!");

	size_t* hostHist = new size_t[arraySize];

	size_t* deviceHist;
	HANDLE_ERROR(cudaMalloc((void**)&deviceHist,sizeof(size_t)*arraySize));
	HANDLE_ERROR(cudaMemset(deviceHist,0,sizeof(size_t)*arraySize));
	DEBUG_KERNEL_CHECK();

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	size_t numValsPerChunk = (size_t)((availMem*MAX_MEM_AVAIL)/sizeof(PixelType));

	PixelType* deviceBuffer;

	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer,sizeof(PixelType)*numValsPerChunk));

	double binSize = (maxVal-minVal)/arraySize;
	int i = 0;
	size_t startIdx = 0;

	while (startIdx<dims.product())
	{
		size_t numValues = MIN(numValsPerChunk,dims.product()-startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer,imageIn+startIdx,sizeof(PixelType)*numValues,cudaMemcpyHostToDevice));

		cudaHistogramCreate<<<props.multiProcessorCount,props.maxThreadsPerBlock,sizeof(size_t)*arraySize>>>(deviceBuffer,
			numValues, deviceHist, minVal, binSize, arraySize);
		DEBUG_KERNEL_CHECK();
		cudaThreadSynchronize();

		++i;
		startIdx = i*numValsPerChunk;
	}

	HANDLE_ERROR(cudaFree(deviceBuffer));

	return deviceHist;
}

template <class PixelType>
size_t* calculateHistogram(const PixelType* imageIn, Vec<size_t> dims, unsigned int arraySize,
						   PixelType minVal=std::numeric_limits<PixelType>::lowest(), PixelType maxVal=std::numeric_limits<PixelType>::max(),
						   int device=0)
{
	size_t* hostHist = new size_t[arraySize];

	size_t* deviceHist = createHistogram(device, arraySize, dims, maxVal, minVal, imageIn);

	HANDLE_ERROR(cudaMemcpy(hostHist,deviceHist,sizeof(size_t)*arraySize,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceHist));

	return hostHist;
}

template <class PixelType>
double* normalizeHistogram(const PixelType* imageIn, Vec<size_t> dims, unsigned int arraySize,
						   PixelType minVal=std::numeric_limits<PixelType>::lowest(), PixelType maxVal=std::numeric_limits<PixelType>::max(),
						   int device=0)
{
	size_t* deviceHist = createHistogram(device, arraySize, dims, maxVal, minVal, imageIn);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	double* hostHistNorm = new double[arraySize];
	double* deviceHistNorm;

	HANDLE_ERROR(cudaMalloc((void**)&deviceHistNorm,sizeof(double)*arraySize));

	int threads = MIN(arraySize,props.maxThreadsPerBlock);
	int blocks = (int)MAX((double)arraySize/threads,(double)props.multiProcessorCount);

	cudaNormalizeHistogram<<<blocks,threads>>>(deviceHist,deviceHistNorm,arraySize,(double)(dims.product()));
	DEBUG_KERNEL_CHECK();

	HANDLE_ERROR(cudaMemcpy(hostHistNorm,deviceHistNorm,sizeof(double)*arraySize,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceHist));
	HANDLE_ERROR(cudaFree(deviceHistNorm));

	return hostHistNorm;
}