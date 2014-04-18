#pragma once
#include "Vec.h"
#include "CudaUtilities.cuh"

template <class PixelType>
__global__ void cudaSum(PixelType* arrayIn, double* arrayOut, size_t n)
{
	extern __shared__ double sums[];

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t stride = blockDim.x*gridDim.x;
	if (i<n)
	{
		sums[threadIdx.x] = (double)(arrayIn[i]);

		while (i<n)
		{
			sums[threadIdx.x] += (double)(arrayIn[i]);

			i += stride;
		}
		__syncthreads();


		for (int reduceUpTo = blockDim.x/2; reduceUpTo>0; reduceUpTo /= 2)
		{
			if (threadIdx.x<reduceUpTo)
				sums[threadIdx.x] += sums[threadIdx.x+reduceUpTo];
			__syncthreads();
		}

		if (threadIdx.x==0)
		{
			arrayOut[blockIdx.x] = sums[0];
		}
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

	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer,sizeof(PixelType)*numValsPerChunk));
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*props.multiProcessorCount));

	hostSum = new double[props.multiProcessorCount];

	for (size_t startIdx=0; startIdx<n; startIdx += numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk,n-startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer,imageIn+startIdx,sizeof(PixelType)*curNumVals,cudaMemcpyHostToDevice));

		int threads = (int)MIN((size_t)props.maxThreadsPerBlock,curNumVals);
		int blocks = MIN(props.multiProcessorCount,(int)ceil((double)curNumVals/threads));

		cudaSum<<<blocks,threads,sizeof(double)*threads>>>(deviceBuffer,deviceSum,curNumVals);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*blocks,cudaMemcpyDeviceToHost));

		for (int i=0; i<blocks; ++i)
		{
			sum += hostSum[i];
		}
	}

	HANDLE_ERROR(cudaFree(deviceSum));
	HANDLE_ERROR(cudaFree(deviceBuffer));

	delete[] hostSum;

	return sum;
}