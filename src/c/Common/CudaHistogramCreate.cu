#include "CudaKernels.cuh"

__global__ void cudaHistogramCreate( CudaImageContainer imageIn, size_t* histogram )
{
	//This code is modified from that of Sanders - Cuda by Example
	__shared__ size_t tempHisto[NUM_BINS];
	tempHisto[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < imageIn.getDeviceDims().product())
	{
		atomicAdd(&(tempHisto[imageIn[i]]), 1);
		i += stride;
	}

	__syncthreads();
	atomicAdd(&(histogram[threadIdx.x]), tempHisto[threadIdx.x]);
}

