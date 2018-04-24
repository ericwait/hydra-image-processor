#pragma once

#include "CudaImageContainer.cuh" 
#include "Vec.h" 

template <class PixelType>
__global__ void cudaGetMinMax(PixelType* arrayIn, PixelType* minsOut, PixelType* maxsOut, size_t n, PixelType minVal, PixelType maxVal)
{
	extern __shared__ unsigned char sharedMem[];
	PixelType* mins = (PixelType*)sharedMem;
	PixelType* maxs = mins + blockDim.x;

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t imStride = blockDim.x*gridDim.x;

	if (i<n)
	{
		mins[threadIdx.x] = arrayIn[i];
		maxs[threadIdx.x] = arrayIn[i];

		while (i + imStride<n)
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


	for (int localStride = blockDim.x / 2; localStride>0; localStride /= 2)
	{
		if (threadIdx.x<localStride)
		{
			if (mins[threadIdx.x] > mins[threadIdx.x + localStride])
				mins[threadIdx.x] = mins[threadIdx.x + localStride];
			if (maxs[threadIdx.x] < maxs[threadIdx.x + localStride])
				maxs[threadIdx.x] = maxs[threadIdx.x + localStride];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		minsOut[blockIdx.x] = mins[0];
		maxsOut[blockIdx.x] = maxs[0];
	}
	__syncthreads();
}

template <typename PixelType>
class MinMaxMem
{
public:
	MinMaxMem()
	{
		initMin = std::numeric_limits<PixelType>::lowest();
		initMax = std::numeric_limits<PixelType>::max();

		deviceMins = NULL;
		deviceMaxs = NULL;
		hostMins = NULL;
		hostMaxs = NULL;

		maxThreads = 0;
		maxBlocks = 0;
	}

	~MinMaxMem()
	{
		HANDLE_ERROR(cudaFree(deviceMins));
		HANDLE_ERROR(cudaFree(deviceMaxs));

		delete[] hostMins;
		delete[] hostMaxs;
	}

	void memalloc(size_t numValsPerChunk, int& threads, int& blocks)
	{
		maxThreads = getKernelMaxThreads(cudaGetMinMax<PixelType>);
		maxBlocks = (int)ceil((double)numValsPerChunk / ((double)maxThreads * 2.0));

		HANDLE_ERROR(cudaMalloc((void**)&deviceMins, sizeof(PixelType)*maxBlocks));
		HANDLE_ERROR(cudaMalloc((void**)&deviceMaxs, sizeof(PixelType)*maxBlocks));

		hostMins = new PixelType[maxBlocks];
		hostMaxs = new PixelType[maxBlocks];

		threads = maxThreads;
		blocks = maxBlocks;
	}

	void retrieve(PixelType& minVal, PixelType& maxVal)
	{
		HANDLE_ERROR(cudaMemcpy(hostMins, deviceMins, sizeof(PixelType)*maxBlocks, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(hostMaxs, deviceMaxs, sizeof(PixelType)*maxBlocks, cudaMemcpyDeviceToHost));

		for (int i = 0; i < maxBlocks; ++i)
		{
			if (minVal > hostMins[i])
				minVal = hostMins[i];
			if (maxVal < hostMaxs[i])
				maxVal = hostMaxs[i];
		}
	}

	PixelType initMin;
	PixelType initMax;
	PixelType* deviceMins;
	PixelType* deviceMaxs;
	PixelType* hostMins;
	PixelType* hostMaxs;

	int maxThreads;
	int maxBlocks;
};

template <class PixelType>
void cGetMinMax(CudaImageContainer<PixelType>* cudaImage, PixelType& minVal, PixelType& maxVal)
{
	minVal = std::numeric_limits<PixelType>::max();
	maxVal = std::numeric_limits<PixelType>::lowest();

	size_t numValsPerChunk = cudaImage->getDims().product();
	int threads, blocks;

	MinMaxMem<PixelType> minMaxMem;
	minMaxMem.memalloc(numValsPerChunk, threads, blocks);

	size_t sharedMemSize = sizeof(PixelType)*threads * 2;
	cudaGetMinMax<<<blocks, threads, sharedMemSize>>>(cudaImage->getImagePointer(), minMaxMem.deviceMins, minMaxMem.deviceMaxs, numValsPerChunk, minMaxMem.initMin, minMaxMem.initMax);
	DEBUG_KERNEL_CHECK();

	minMaxMem.retrieve(minVal, maxVal);
}

template <class PixelType>
void cGetMinMax(const PixelType* imageIn, size_t numValues, PixelType& minVal, PixelType& maxVal, int device = 0)
{
	cudaSetDevice(device);

	minVal = std::numeric_limits<PixelType>::max();
	maxVal = std::numeric_limits<PixelType>::lowest();
	PixelType* deviceBuffer = NULL;
	MinMaxMem<PixelType> minMaxMem;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem, &total);

	size_t numValsPerChunk = MIN(numValues, (size_t)((availMem*MAX_MEM_AVAIL) / sizeof(PixelType)));
	int threads, maxBlocks;
	minMaxMem.memalloc(numValsPerChunk, threads, maxBlocks);
	HANDLE_ERROR(cudaMalloc((void**)&deviceBuffer, sizeof(PixelType)*numValsPerChunk));

	for (size_t startIdx = 0; startIdx < numValues; startIdx += numValsPerChunk)
	{
		size_t curNumVals = MIN(numValsPerChunk, numValues - startIdx);

		HANDLE_ERROR(cudaMemcpy(deviceBuffer, imageIn + startIdx, sizeof(PixelType)*curNumVals, cudaMemcpyHostToDevice));

		int blocks = (int)((double)curNumVals / (threads * 2));
		size_t sharedMemSize = sizeof(PixelType)*threads * 2;

		cudaGetMinMax<<<blocks, threads, sharedMemSize>>>(deviceBuffer, minMaxMem.deviceMins, minMaxMem.deviceMaxs, curNumVals, minMaxMem.initMin, minMaxMem.initMax);
		DEBUG_KERNEL_CHECK();

		minMaxMem.retrieve(minVal, maxVal);
	}

	HANDLE_ERROR(cudaFree(deviceBuffer));
}
