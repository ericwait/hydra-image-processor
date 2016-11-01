#pragma once
#include "CudaImageContainer.cuh"
#include "Vec.h"

template <class PixelType>
__global__ void cudaGetMinMax(PixelType* arrayIn, PixelType* minsOut, PixelType* maxsOut, size_t n, PixelType minVal, PixelType maxVal)
{
	extern __shared__ unsigned char sharedMem[];
	PixelType* mins = (PixelType*)sharedMem;
	PixelType* maxs = mins+blockDim.x;

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t imStride = blockDim.x*gridDim.x;

	if (i<n)
	{
		mins[threadIdx.x] = arrayIn[i];
		maxs[threadIdx.x] = arrayIn[i];

		while (i+imStride<n)
		{
			i += imStride;

			if (mins[threadIdx.x] > arrayIn[i])
				mins[threadIdx.x] = arrayIn[i];
			if (maxs[threadIdx.x] < arrayIn[i])
				maxs[threadIdx.x] = arrayIn[i];
		}
	}
	else
	{
		mins[threadIdx.x] = maxVal;
		maxs[threadIdx.x] = minVal;
	}

	__syncthreads();


	for (int localStride=blockDim.x/2; localStride>0; localStride /= 2)
	{
		if (threadIdx.x<localStride)
		{
			if (mins[threadIdx.x] > mins[threadIdx.x+localStride])
				mins[threadIdx.x] = mins[threadIdx.x+localStride];
			if (maxs[threadIdx.x] < maxs[threadIdx.x+localStride])
				maxs[threadIdx.x] = maxs[threadIdx.x+localStride];
		}

		__syncthreads();
	}

	if (threadIdx.x==0)
	{
		minsOut[blockIdx.x] = mins[0];
		maxsOut[blockIdx.x] = maxs[0];
	}
	__syncthreads();
}

template <class PixelType>
void cGetMinMax(const PixelType* imageIn, Vec<size_t> dims, PixelType& minVal, PixelType& maxVal, int device=0)
{
    cudaSetDevice(device);

	size_t n = dims.product();
	minVal = std::numeric_limits<PixelType>::max();
	maxVal = std::numeric_limits<PixelType>::lowest();
	PixelType initMin= std::numeric_limits<PixelType>::lowest();
	PixelType initMax = std::numeric_limits<PixelType>::max();
	PixelType* deviceMins;
	PixelType* deviceMaxs;
	PixelType* hostMins;
	PixelType* hostMaxs;
	PixelType* deviceBuffer;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	size_t numValsPerChunk = MIN(n,(size_t)((availMem*MAX_MEM_AVAIL)/sizeof(PixelType)));

	int threads = props.maxThreadsPerBlock;
	int maxBlocks = (int)ceil((double)numValsPerChunk/(threads*2)); 

	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer,sizeof(PixelType)*numValsPerChunk));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMins,sizeof(PixelType)*maxBlocks));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMaxs,sizeof(PixelType)*maxBlocks));

	hostMins = new PixelType[maxBlocks];
	hostMaxs = new PixelType[maxBlocks];

	for (size_t startIdx=0; startIdx<n; startIdx += numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk,n-startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer,imageIn+startIdx,sizeof(PixelType)*curNumVals,cudaMemcpyHostToDevice));

		int blocks = (int)((double)curNumVals/(threads*2));
		size_t sharedMemSize = sizeof(PixelType)*threads*2;

		cudaGetMinMax<<<blocks,threads,sharedMemSize>>>(deviceBuffer,deviceMins,deviceMaxs,curNumVals,initMin,initMax);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostMins,deviceMins,sizeof(PixelType)*blocks,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(hostMaxs,deviceMaxs,sizeof(PixelType)*blocks,cudaMemcpyDeviceToHost));

		for (int i=0; i<blocks; ++i)
		{
			if (minVal > hostMins[i])
				minVal = hostMins[i];
			if (maxVal < hostMaxs[i])
				maxVal = hostMaxs[i];
		}
	}

	HANDLE_ERROR(cudaFree(deviceMins));
	HANDLE_ERROR(cudaFree(deviceMaxs));
	HANDLE_ERROR(cudaFree(deviceBuffer));

	delete[] hostMins;
	delete[] hostMaxs;
}