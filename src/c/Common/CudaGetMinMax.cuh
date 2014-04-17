#pragma once
#include "CudaImageContainer.cuh"
#include "Vec.h"

template <class PixelType>
__global__ void cudaGetMinMax(PixelType* arrayIn, double* minsOut, double* maxsOut, size_t n)
{
	extern __shared__ double mins[];
	double* maxs = mins+blockDim.x;

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t stride = blockDim.x*gridDim.x;
	if (i<n)
	{
		mins[threadIdx.x] = (double)(arrayIn[i]);
		maxs[threadIdx.x] = (double)(arrayIn[i]);

		while (i<n)
		{
			if (mins[threadIdx.x] > (double)(arrayIn[i]))
				mins[threadIdx.x] = (double)(arrayIn[i]);
			if (maxs[threadIdx.x] < (double)(arrayIn[i]))
				maxs[threadIdx.x] = (double)(arrayIn[i]);

			i += stride;
		}
		__syncthreads();


		for (int reduceUpTo = blockDim.x/2; reduceUpTo>0; reduceUpTo /= 2)
		{
			if (threadIdx.x<reduceUpTo)
			{
				if (mins[threadIdx.x] > mins[threadIdx.x+reduceUpTo])
					mins[threadIdx.x] = mins[threadIdx.x+reduceUpTo];
				if (maxs[threadIdx.x] < maxs[threadIdx.x+reduceUpTo])
					maxs[threadIdx.x] = maxs[threadIdx.x+reduceUpTo];
			}else 
				break;
			__syncthreads();
		}

		if (threadIdx.x==0)
		{
			minsOut[blockIdx.x] = mins[0];
			maxsOut[blockIdx.x] = maxs[0];
		}
	}
	__syncthreads();
}

template <class PixelType>
void getMinMax(const PixelType* imageIn, Vec<size_t> dims, PixelType& minVal, PixelType& maxVal, int device=0)
{
	size_t n = dims.product();
	minVal = std::numeric_limits<PixelType>::max();
	maxVal = std::numeric_limits<PixelType>::lowest();
	double* deviceMins;
	double* deviceMaxs;
	double* hostMins;
	double* hostMaxs;
	PixelType* deviceBuffer;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	size_t numValsPerChunk = MIN(n,(size_t)((availMem*MAX_MEM_AVAIL)/sizeof(PixelType)));

	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer,sizeof(PixelType)*numValsPerChunk));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMins,sizeof(double)*props.multiProcessorCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMaxs,sizeof(double)*props.multiProcessorCount));

	hostMins = new double[props.multiProcessorCount];
	hostMaxs = new double[props.multiProcessorCount];

	for (size_t startIdx=0; startIdx<n; startIdx += numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk,n-startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer,imageIn+startIdx,sizeof(PixelType)*curNumVals,cudaMemcpyHostToDevice));

		int threads = (int)MIN((size_t)props.maxThreadsPerBlock,curNumVals);
		int blocks = MIN(props.multiProcessorCount,(int)ceil((double)curNumVals/threads));

		cudaGetMinMax<<<blocks,threads,sizeof(double)*threads*2>>>(deviceBuffer,deviceMins,deviceMaxs,curNumVals);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostMins,deviceMins,sizeof(double)*blocks,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(hostMaxs,deviceMaxs,sizeof(double)*blocks,cudaMemcpyDeviceToHost));

		for (int i=0; i<blocks; ++i)
		{
			if (minVal > (PixelType)hostMins[i])
				minVal = (PixelType)hostMins[i];
			if (maxVal < (PixelType)hostMaxs[i])
				maxVal = (PixelType)hostMaxs[i];
		}
	}

	HANDLE_ERROR(cudaFree(deviceMins));
	HANDLE_ERROR(cudaFree(deviceMaxs));
	HANDLE_ERROR(cudaFree(deviceBuffer));

	delete[] hostMins;
	delete[] hostMaxs;
}