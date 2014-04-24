#pragma once
#include "Vec.h"
#include "CudaUtilities.cuh"
#include "cuda_runtime.h"

template <class PixelType>
__global__ void cudaSum(PixelType* arrayIn, double* arrayOut, size_t n)
{
	extern __shared__ double sums[];

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t imStride = blockDim.x*gridDim.x;

	if (i<n)
	{
		sums[threadIdx.x] = (double)(arrayIn[i]);
		while (i+imStride<n)
		{
			sums[threadIdx.x] += (double)(arrayIn[i+imStride]);
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
		else
			break;
		__syncthreads();
	}

	if (threadIdx.x==0)
	{
		arrayOut[blockIdx.x] = sums[0];
	}
	__syncthreads();
}

template <class PixelType>
double sumArray(const PixelType* imageIn, size_t n, int device=0)
{
	double sum = 0.0;
	double* deviceSum;
	double* hostSum;
	PixelType* deviceBuffer;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	size_t numValsPerChunk = MIN(n,(size_t)((availMem*MAX_MEM_AVAIL)/sizeof(PixelType)));

	int maxBlocks = (int)ceil((double)numValsPerChunk/(2*props.maxThreadsPerBlock)); 
	int threads = props.maxThreadsPerBlock;

	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer,sizeof(PixelType)*numValsPerChunk));
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*maxBlocks));

	hostSum = new double[maxBlocks];

	for (size_t startIdx=0; startIdx<n; startIdx += numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk,n-startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer,imageIn+startIdx,sizeof(PixelType)*curNumVals,cudaMemcpyHostToDevice));

		int blocks = (int)ceil((double)curNumVals/(2*props.maxThreadsPerBlock));
		size_t sharedMemSize = sizeof(double)*props.maxThreadsPerBlock;

		cudaSum<<<blocks,threads,sharedMemSize>>>(deviceBuffer,deviceSum,curNumVals);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*blocks,cudaMemcpyDeviceToHost));

		for (int i=0; i<blocks; ++i)
		{
			sum += hostSum[i];
		}

		memset(hostSum,0,sizeof(double)*maxBlocks);
	}

	HANDLE_ERROR(cudaFree(deviceSum));
	HANDLE_ERROR(cudaFree(deviceBuffer));

	delete[] hostSum;

	return sum;
}