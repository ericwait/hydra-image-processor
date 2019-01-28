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
#include <cstring>
#include <limits>
#include <omp.h>

template <class PixelTypeIn, class OutType>
__global__ void cudaSum(const PixelTypeIn* arrayIn, OutType* arrayOut, std::size_t n)
{
	extern __shared__ unsigned char sharedMem[];
	OutType* sums = (OutType*)sharedMem;

	std::size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	std::size_t imStride = blockDim.x*gridDim.x;

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

template <class PixelTypeIn, class OutType>
void sumBuffer(ImageChunk &chunk, CudaImageContainer<PixelTypeIn>* buffer, OutType& sum, OutType* deviceSum=NULL, OutType* hostSum=NULL)
{
	int blocks = chunk.blocks.x*chunk.blocks.y*chunk.blocks.z;
	blocks = (int)(ceil((double)blocks / 2.0));
	int threads = chunk.threads.x*chunk.threads.y*chunk.threads.z;

	bool cleanGPU = false;
	if (NULL == deviceSum)
	{
		HANDLE_ERROR(cudaMalloc((void**)&deviceSum, sizeof(OutType)*blocks));
		cleanGPU = true;
	}
	bool cleanCPU = false;
	if (NULL == hostSum)
	{
		hostSum = new OutType[blocks];
		cleanCPU = true;
	}

	std::size_t sharedMemSize = threads*sizeof(OutType);

	HANDLE_ERROR(cudaMemset(deviceSum, 0, sizeof(OutType)*blocks));
	std::memset(hostSum, 0, sizeof(OutType)*blocks);

	cudaSum<<<blocks, threads, sharedMemSize>>>(buffer->getConstImagePointer(), deviceSum, chunk.getFullChunkSize().product());

	DEBUG_KERNEL_CHECK();
	HANDLE_ERROR(cudaMemcpy(hostSum, deviceSum, sizeof(OutType)*blocks, cudaMemcpyDeviceToHost));

	for (int i = 0; i < blocks; ++i)
	{
		sum += hostSum[i];
	}

	if (cleanGPU)
	{
		HANDLE_ERROR(cudaFree(deviceSum));
		deviceSum = NULL;
	}

	if (!cleanCPU)
		std::memset(hostSum, 0, sizeof(OutType)*blocks);
	else
	{
		delete[] hostSum;
		hostSum = NULL;
	}
}

template <class OutType, class PixelTypeIn>
void cSum(ImageView<PixelTypeIn> imageIn, OutType& outVal, int device=-1)
{
	const int NUM_BUFF_NEEDED = 1;
	outVal = 0;
	
	CudaDevices cudaDevs(cudaSum<PixelTypeIn, OutType>, device);

	if (imageIn.getNumElements() <= cudaDevs.getMaxThreadsPerBlock())
	{
		const PixelTypeIn* imPtr = imageIn.getConstPtr();
		for (std::size_t i = 0; i <imageIn.getNumElements(); ++i)
		{
			outVal += imPtr[i];
		}
		return;
	}

	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, sizeof(PixelTypeIn));

	Vec<std::size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	int numThreads = MIN(chunks.size(), cudaDevs.getNumDevices());
	OutType* sums = new OutType[numThreads];
	for (int i = 0; i < numThreads; ++i)
		sums[i] = 0;

	omp_set_num_threads(numThreads);
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeIn> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		sums[CUDA_IDX] = 0;
		OutType* deviceSum;

		int maxBlocks = (int)ceil((double)chunks[0].getFullChunkSize().product() / (chunks[0].threads.x*chunks[0].threads.y*chunks[0].threads.z * 2));

		HANDLE_ERROR(cudaMalloc((void**)&deviceSum, sizeof(OutType)*maxBlocks));
		OutType* hostSum = new OutType[maxBlocks];

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			sumBuffer(chunks[i], deviceImages.getCurBuffer(), sums[CUDA_IDX], deviceSum, hostSum);
		}

		HANDLE_ERROR(cudaFree(deviceSum));
		delete[] hostSum;
	}

	for (int i = 0; i < numThreads; ++i)
		outVal += sums[i];

	delete[] sums;
}
