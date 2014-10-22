#pragma once
#include "Vec.h"
#include "CudaUtilities.cuh"
#include "cuda_runtime.h"

template <class PixelTypeIn, class OutType>
__global__ void cudaSum(const PixelTypeIn* arrayIn, OutType* arrayOut, size_t n)
{
	extern __shared__ unsigned char sharedMem[];
	OutType* sums = (OutType*)sharedMem;

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t imStride = blockDim.x*gridDim.x;

	if (i<n)
	{
		sums[threadIdx.x] = (OutType)(arrayIn[i]);
		while (i+imStride<n)
		{
			sums[threadIdx.x] += (OutType)(arrayIn[i+imStride]);
			i += imStride;
		}
	}
	else
	{
		sums[threadIdx.x] = 0;
	}

	__syncthreads();

	for (int localStride=blockDim.x/2; localStride>0; localStride=localStride/2)
	{
		if (threadIdx.x<localStride)
			sums[threadIdx.x] += sums[threadIdx.x+localStride];

		__syncthreads();
	}

	if (threadIdx.x==0)
	{
		arrayOut[blockIdx.x] = sums[0];
	}
	__syncthreads();
}

template < class OutType, class PixelTypeIn>
OutType cSumArray(const PixelTypeIn* imageIn, size_t n, int device=0)
{
	OutType sum = 0;
	OutType* deviceSum;
	OutType* hostSum;
	PixelTypeIn* deviceBuffer;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	if (n <= props.maxThreadsPerBlock)
	{
		for (size_t i = 0; i < n; ++i)
		{
			sum += imageIn[i];
		}

		return sum;
	}

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	size_t numValsPerChunk = MIN(n,(size_t)((availMem*MAX_MEM_AVAIL)/sizeof(PixelTypeIn)));

	int threads = props.maxThreadsPerBlock;
	int maxBlocks = (int)ceil((double)numValsPerChunk/(threads*2)); 

	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer,sizeof(PixelTypeIn)*numValsPerChunk));
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(OutType)*maxBlocks));

	hostSum = new OutType[maxBlocks];

	for (size_t startIdx=0; startIdx<n; startIdx += numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk,n-startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer,imageIn+startIdx,sizeof(PixelTypeIn)*curNumVals,cudaMemcpyHostToDevice));

		int blocks = (int)ceil((double)curNumVals/(threads*2));
		size_t sharedMemSize = sizeof(OutType)*threads;

		cudaSum<<<blocks,threads,sharedMemSize>>>(deviceBuffer,deviceSum,curNumVals);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(OutType)*blocks,cudaMemcpyDeviceToHost));

		for (int i=0; i<blocks; ++i)
		{
			sum += hostSum[i];
		}

		memset(hostSum,0,sizeof(OutType)*maxBlocks);
	}

	HANDLE_ERROR(cudaFree(deviceSum));
	HANDLE_ERROR(cudaFree(deviceBuffer));

	delete[] hostSum;

	return sum;
}