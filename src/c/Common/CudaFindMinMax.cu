#include "CudaKernels.cuh"

__global__ void cudaFindMinMax( CudaImageContainer arrayIn, double* minArrayOut, double* maxArrayOut, size_t n )
{
	extern __shared__ double maxData[];
	extern __shared__ double minData[];

	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockDim.x*2 + tid;
	size_t gridSize = blockDim.x*2*gridDim.x;

	while (i<n)
	{
		maxData[tid] = arrayIn[i];
		minData[tid] = arrayIn[i];

		if (i+blockDim.x<n)
		{
			if(maxData[tid]<arrayIn[i+blockDim.x])
				maxData[tid] = arrayIn[i+blockDim.x];

			if(minData[tid]>arrayIn[i+blockDim.x])
				minData[tid] = arrayIn[i+blockDim.x];
		}

		i += gridSize;
	}
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
		{
			if(maxData[tid]<maxData[tid + 1024])
				maxData[tid] = maxData[tid + 1024];

			if(minData[tid]>minData[tid + 1024])
				minData[tid] = minData[tid + 1024];
		}
		__syncthreads();
	}
	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
		{
			if(maxData[tid]<maxData[tid + 512])
				maxData[tid] = maxData[tid + 512];

			if(minData[tid]>minData[tid + 512])
				minData[tid] = minData[tid + 512];
		}
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) 
		{
			if(maxData[tid]<maxData[tid + 256])
				maxData[tid] = maxData[tid + 256];

			if(minData[tid]>minData[tid + 256])
				minData[tid] = minData[tid + 256];
		}
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
		{
			if(maxData[tid]<maxData[tid + 128])
				maxData[tid] = maxData[tid + 128];

			if(minData[tid]>minData[tid + 128])
				minData[tid] = minData[tid + 128];
		}
		__syncthreads(); 
	}
	if (blockDim.x >= 128) 
	{
		if (tid < 64)
		{
			if(maxData[tid]<maxData[tid + 64])
				maxData[tid] = maxData[tid + 64];

			if(minData[tid]>minData[tid + 64])
				minData[tid] = minData[tid + 64];
		}
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			{
				if(maxData[tid]<maxData[tid + 64])
					maxData[tid] = maxData[tid + 64];

				if(minData[tid]>minData[tid + 64])
					minData[tid] = minData[tid + 64];
			}
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			if(maxData[tid]<maxData[tid + 16])
				maxData[tid] = maxData[tid + 16];

			if(minData[tid]>minData[tid + 16])
				minData[tid] = minData[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			if(maxData[tid]<maxData[tid + 8])
				maxData[tid] = maxData[tid + 8];

			if(minData[tid]>minData[tid + 8])
				minData[tid] = minData[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			if(maxData[tid]<maxData[tid + 4])
				maxData[tid] = maxData[tid + 4];

			if(minData[tid]>minData[tid + 4])
				minData[tid] = minData[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			if(maxData[tid]<maxData[tid + 2])
				maxData[tid] = maxData[tid + 2];

			if(minData[tid]>minData[tid + 2])
				minData[tid] = minData[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			if(maxData[tid]<maxData[tid + 1])
				maxData[tid] = maxData[tid + 1];

			if(minData[tid]>minData[tid + 1])
				minData[tid] = minData[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
	{
		minArrayOut[blockIdx.x] = minData[0];
		maxArrayOut[blockIdx.x] = maxData[0];
	}
}

