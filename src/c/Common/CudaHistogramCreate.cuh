#pragma once
#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaHistogramCreate( CudaImageContainer<PixelType> imageIn, size_t* histogram )
{
	//This code is modified from that of Sanders - Cuda by Example
	__shared__ size_t tempHisto[NUM_BINS];
	tempHisto[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < imageIn.getDeviceDims().product())
	{
		atomicAdd(&(tempHisto[(int)imageIn[i]]), 1);
		i += stride;
	}

	__syncthreads();
	atomicAdd(&(histogram[threadIdx.x]), tempHisto[threadIdx.x]);
}

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, Vec<size_t> imageDims)
{
	int x = blockIdx.x;
	normHistogram[x] = (double)(histogram[x]) / (imageDims.x*imageDims.y*imageDims.z);
}
