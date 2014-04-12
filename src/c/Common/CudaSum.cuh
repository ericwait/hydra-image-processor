#pragma once
#include "Vec.h"
#include "CudaUtilities.cuh"

template <class PixelType>
__global__ void cudaSum(PixelType* arrayIn, double* arrayOut, size_t n)
{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013

	extern __shared__ double sdata[];

	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockDim.x + tid;
	size_t gridSize = blockDim.x*gridDim.x;
	sdata[tid] = (double)(arrayIn[i]);

	do
	{
		if (i+blockDim.x<n)
			sdata[tid] += (double)(arrayIn[i+blockDim.x]);

		i += gridSize;
	}while (i<n);
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
			sdata[tid] += sdata[tid + 1024];
		__syncthreads();
	}

	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}

	if (blockDim.x >= 512)
	{
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}

	if (blockDim.x >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads(); 
	}

	if (blockDim.x >= 128) 
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			sdata[tid] += sdata[tid + 32];
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			sdata[tid] += sdata[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			sdata[tid] += sdata[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			sdata[tid] += sdata[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			sdata[tid] += sdata[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			sdata[tid] += sdata[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
		arrayOut[blockIdx.x] = sdata[0];
}

template <class PixelType>
double sumArray(const PixelType* imageIn, size_t n, int device=0)
{
	double sum = 0.0;
	double* deviceSum;
	double* hostSum;
	PixelType* deviceImage;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	unsigned int blocks = props.multiProcessorCount;
	unsigned int threads = props.maxThreadsPerBlock;

	Vec<size_t> maxDeviceDims(1,1,1);

	maxDeviceDims.x = (n < availMem*MAX_MEM_AVAIL/sizeof(PixelType)) ? (n) :
		((size_t)(availMem*MAX_MEM_AVAIL/sizeof(PixelType)));

	HANDLE_ERROR(cudaMalloc((void**)&deviceImage,sizeof(PixelType)*maxDeviceDims.x));
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*blocks));
	hostSum = new double[blocks];

	for (int i=0; i<ceil((double)n/maxDeviceDims.x); ++i)
	{
		const PixelType* imStart = imageIn + i*maxDeviceDims.x;
		size_t numValues = ((i+1)*maxDeviceDims.x < n) ? (maxDeviceDims.x) : (n-i*maxDeviceDims.x);

		HANDLE_ERROR(cudaMemcpy(deviceImage,imStart,sizeof(PixelType)*numValues,cudaMemcpyHostToDevice));

		cudaSum<<<blocks,threads,sizeof(double)*threads>>>(deviceImage,deviceSum,numValues);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*blocks,cudaMemcpyDeviceToHost));

		for (unsigned int i=0; i<blocks; ++i)
		{
			sum += hostSum[i];
		}
	}

	HANDLE_ERROR(cudaFree(deviceSum));
	HANDLE_ERROR(cudaFree(deviceImage));

	delete[] hostSum;

	return sum;
}