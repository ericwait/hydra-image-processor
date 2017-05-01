#pragma once

#include "Vec.h"
#include <vector>
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaMultAddFilter.cuh"
#include "CudaUtilities.cuh"
#include "CudaConvertType.cuh"
#include <float.h>

template <class PixelType>
void runLoGIterations(Vec<int> &gaussIterations, std::vector<ImageChunk>::iterator& curChunk, CudaDeviceImages<PixelType>& deviceImages,
						Vec<size_t> sizeconstKernelDims, int device)
{
	cudaSetDevice(device);
	if(curChunk->getFullChunkSize().x>1)
	{
		for(int x = 0; x<gaussIterations.x; ++x)
		{
			cudaMultAddFilter<<<curChunk->blocks, curChunk->threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()),
																	   Vec<size_t>(sizeconstKernelDims.x, 1, 1),0, false);
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();
		}
	}

	if(curChunk->getFullChunkSize().y>1)
	{
		for(int y = 0; y<gaussIterations.y; ++y)
		{
			cudaMultAddFilter<<<curChunk->blocks, curChunk->threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()),
																	   Vec<size_t>(1, sizeconstKernelDims.y, 1), sizeconstKernelDims.x, false);
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();
		}
	}

	if(curChunk->getFullChunkSize().z>1)
	{
		for(int z = 0; z<gaussIterations.z; ++z)
		{
			cudaMultAddFilter<<<curChunk->blocks, curChunk->threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()),
																	   Vec<size_t>(1, 1, sizeconstKernelDims.z), sizeconstKernelDims.y+sizeconstKernelDims.x, false);
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();
		}
	}
}

template <class PixelType>
float* cLoGFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas, float** imageOut = NULL, int device = 0)
{
	cudaSetDevice(device);
	float* imOut = setUpOutIm(dims, imageOut);

	Vec<int> loGIterations(0, 0, 0);
	sigmas.x = (dims.x==1) ? (0) : (sigmas.x);
	sigmas.y = (dims.y==1) ? (0) : (sigmas.y);
	sigmas.z = (dims.z==1) ? (0) : (sigmas.z);

	float* hostKernel = NULL;

	Vec<size_t> sizeconstKernelDims = createLoGKernel(sigmas, &hostKernel, loGIterations);
	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)* sizeconstKernelDims.sum()));
		
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem, &total);

	int blockSize = getKernelMaxThreads(cudaMultAddFilter<float>);
	float memoryDem = sizeof(float)*2+sizeof(PixelType);
	float memoryForFloat = sizeof(float)*2 / memoryDem;
	float memoryInput = sizeof(PixelType)/memoryDem;

	if(memoryInput<1.0f/3.0f+1e-4 && memoryInput>1.0f/3.0f-1e-4)
	{
		// This is when the input is already a float and we can just split up the memory 
		std::vector<ImageChunk> chunks = calculateBuffers<float>(dims, 2, (size_t)(availMem*MAX_MEM_AVAIL), props, sizeconstKernelDims, blockSize);

		Vec<size_t> maxDeviceDims;
		setMaxDeviceDims(chunks, maxDeviceDims);

		CudaDeviceImages<float> deviceImages(2, maxDeviceDims, device);

		for(std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk!=chunks.end(); ++curChunk)
		{
			deviceImages.setAllDims(curChunk->getFullChunkSize());

			curChunk->sendROI(imageIn, dims, deviceImages.getCurBuffer());

			runLoGIterations(loGIterations, curChunk, deviceImages, sizeconstKernelDims, device);

			curChunk->retriveROI(imOut, dims, deviceImages.getCurBuffer());
		}
	}
	else
	{
		// in this case we have to convert the input before running the kernel
		std::vector<ImageChunk> chunks = calculateBuffers<float>(dims, 2, (size_t)(availMem*MAX_MEM_AVAIL*memoryForFloat), props, sizeconstKernelDims, blockSize);

		Vec<size_t> maxDeviceDims;
		setMaxDeviceDims(chunks, maxDeviceDims);

		CudaDeviceImages<PixelType> deviceInputImages(1, maxDeviceDims, device);
		CudaDeviceImages<float> deviceFloatImages(2, maxDeviceDims, device);

		for(std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk!=chunks.end(); ++curChunk)
		{
			deviceInputImages.setAllDims(curChunk->getFullChunkSize());
			deviceFloatImages.setAllDims(curChunk->getFullChunkSize());

			curChunk->sendROI(imageIn, dims, deviceInputImages.getCurBuffer());

			double numVoxels = curChunk->getFullChunkSize().product();
			int numBlocks = ceil(numVoxels/props.maxThreadsPerBlock);
			cudaConvertType<<<numBlocks,props.maxThreadsPerBlock>>>(deviceInputImages.getCurBuffer()->getDeviceImagePointer(), deviceFloatImages.getCurBuffer()->getDeviceImagePointer(), numVoxels, -FLT_MAX, FLT_MAX);

			runLoGIterations(loGIterations, curChunk, deviceFloatImages, sizeconstKernelDims, device);

			curChunk->retriveROI(imOut, dims, deviceFloatImages.getCurBuffer());
		}
	}


	delete[] hostKernel;

	return imOut;
}
