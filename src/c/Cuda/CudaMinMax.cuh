#pragma once
#include "CudaImageContainer.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaUtilities.h"
#include "CudaDeviceInfo.h"
#include "ImageDimensions.cuh"
#include "ImageChunk.h"
#include "Defines.h"
#include "Vec.h"

#include <cuda_runtime.h>
#include <limits>
#include <omp.h>

template <class PixelType>
__global__ void cudaMinMax(const PixelType* arrayIn, PixelType* minsOut, PixelType* maxsOut, size_t n, const PixelType MIN_VAL, const PixelType MAX_VAL)
{
	extern __shared__ unsigned char sharedMem[];
	PixelType* mins = (PixelType*)sharedMem;
	PixelType* maxs = mins + blockDim.x;

	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	size_t imStride = blockDim.x*gridDim.x;

	if (i < n)
	{
		mins[threadIdx.x] = (PixelType)(arrayIn[i]);
		maxs[threadIdx.x] = (PixelType)(arrayIn[i]);
		while (i + imStride < n)
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
		mins[threadIdx.x] = MAX_VAL;
		maxs[threadIdx.x] = MIN_VAL;
	}

	__syncthreads();

	for (int localStride = blockDim.x / 2; localStride > 0; localStride /= 2)
	{
		if (threadIdx.x < localStride)
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

template <class PixelType>
void minMaxBuffer(ImageChunk &chunk, CudaImageContainer<PixelType>* buffer, PixelType& minVal, PixelType& maxVal, size_t maxShared, PixelType* deviceMin = NULL, PixelType* deviceMax = NULL, PixelType* hostMin = NULL, PixelType* hostMax = NULL)
{
	const PixelType MIN_VAL = std::numeric_limits<PixelType>::lowest();
	const PixelType MAX_VAL = std::numeric_limits<PixelType>::max();

	int blocks = chunk.blocks.x*chunk.blocks.y*chunk.blocks.z;
	blocks = (int)(ceil((double)blocks / 2.0));
	int threads = chunk.threads.x*chunk.threads.y*chunk.threads.z;
	size_t sharedMemSize = threads * sizeof(PixelType) * 2;

	minVal = MAX_VAL;
	maxVal = MIN_VAL;

	bool cleanGPU = false;
	if (NULL == deviceMin)
	{
		HANDLE_ERROR(cudaMalloc((void**)&deviceMin, sizeof(PixelType)*blocks));
		cleanGPU = true;
	}
	if (NULL == deviceMax)
	{
		HANDLE_ERROR(cudaMalloc((void**)&deviceMax, sizeof(PixelType)*blocks));
		cleanGPU = true;
	}
	bool cleanCPU = false;
	if (NULL == hostMin)
	{
		hostMin = new PixelType[blocks];
		cleanCPU = true;
	}
	if (NULL == hostMax)
	{
		hostMax = new PixelType[blocks];
		cleanCPU = true;
	}

	cudaMinMax<<<blocks, threads, sharedMemSize>>>(buffer->getConstImagePointer(), deviceMin, deviceMax, chunk.getFullChunkSize().product(),MIN_VAL,MAX_VAL);

	DEBUG_KERNEL_CHECK();
	HANDLE_ERROR(cudaMemcpy(hostMin, deviceMin, sizeof(PixelType)*blocks, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(hostMax, deviceMax, sizeof(PixelType)*blocks, cudaMemcpyDeviceToHost));

	for (int i = 0; i < blocks; ++i)
	{
		if (minVal > hostMin[i])
			minVal = hostMin[i];

		if (maxVal < hostMax[i])
			maxVal = hostMax[i];
	}

	if (cleanGPU)
	{
		HANDLE_ERROR(cudaFree(deviceMin));
		deviceMin = NULL;
		HANDLE_ERROR(cudaFree(deviceMax));
		deviceMax = NULL;
	}

	if (!cleanCPU)
	{
		memset(hostMin, 0, sizeof(PixelType)*blocks);
		memset(hostMax, 0, sizeof(PixelType)*blocks);
	}
	else
	{
		delete[] hostMin;
		delete[] hostMax;
		hostMin = NULL;
		hostMax = NULL;
	}
}

template <class PixelType>
void cMinMax(ImageContainer<PixelType> imageIn, PixelType& outMin, PixelType& outMax, int device = -1)
{
	const int NUM_BUFF_NEEDED = 1;
	const PixelType MIN_VAL = std::numeric_limits<PixelType>::lowest();
	const PixelType MAX_VAL = std::numeric_limits<PixelType>::max();
	
	outMin = MAX_VAL;
	outMax = MIN_VAL;

	CudaDevices cudaDevs(cudaMinMax<PixelType>, device);

	if (imageIn.getNumElements() <= cudaDevs.getMaxThreadsPerBlock())
	{
		const PixelType* imPtr = imageIn.getConstPtr();
		for (size_t i = 0; i < imageIn.getNumElements(); ++i)
		{
			if (outMin > imPtr[i])
				outMin = imPtr[i];

			if (outMax < imPtr[i])
				outMax = imPtr[i];
		}
		return;
	}

	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, sizeof(PixelType));

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	int numThreads = MIN(chunks.size(), cudaDevs.getNumDevices());
	PixelType* mins = new PixelType[numThreads];
	PixelType* maxs = new PixelType[numThreads];
	for (int i = 0; i < numThreads; ++i)
	{
		mins[i] = MAX_VAL;
		maxs[i] = MIN_VAL;
	}

	omp_set_num_threads(numThreads);
#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelType> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		PixelType* deviceMin;
		PixelType* deviceMax;

		int maxBlocks = (int)ceil((double)(chunks[0].blocks.x*chunks[0].blocks.y*chunks[0].blocks.z) / 2.0);

		HANDLE_ERROR(cudaMalloc((void**)&deviceMin, sizeof(PixelType)*maxBlocks));
		HANDLE_ERROR(cudaMalloc((void**)&deviceMax, sizeof(PixelType)*maxBlocks));
		PixelType* hostMin = new PixelType[maxBlocks];
		PixelType* hostMax = new PixelType[maxBlocks];

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			minMaxBuffer(chunks[i], deviceImages.getCurBuffer(), mins[CUDA_IDX], maxs[CUDA_IDX], cudaDevs.getMinSharedMem(), deviceMin, deviceMax, hostMin, hostMax);
		}

		HANDLE_ERROR(cudaFree(deviceMin));
		HANDLE_ERROR(cudaFree(deviceMax));
		delete[] hostMin;
		delete[] hostMax;
	}

	for (int i = 0; i < numThreads; ++i)
	{
		if (outMin > mins[i])
			outMin = mins[i];

		if (outMax < maxs[i])
			outMax = maxs[i];
	}

	delete[] mins;
	delete[] maxs;
}
