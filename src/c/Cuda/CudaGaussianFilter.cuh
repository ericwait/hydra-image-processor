#pragma once

#include "Vec.h"
#include <vector>
#include "ImageChunk.h"
#include "CudaDeviceImages.cuh"
#include "CudaMultAddFilter.cuh"
#include "CudaUtilities.cuh"

template <class PixelType>
void runGaussIterations(Vec<int> &gaussIterations, std::vector<ImageChunk>::iterator& curChunk, CudaDeviceImages<PixelType>& deviceImages,
						Vec<size_t> sizeconstKernelDims, int device)
{
    cudaSetDevice(device);
	if (curChunk->getFullChunkSize().x>1)
	{
		for (int x=0; x<gaussIterations.x; ++x)
		{
			cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
				Vec<size_t>(sizeconstKernelDims.x,1,1));
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();
		}
	}

	if (curChunk->getFullChunkSize().y>1)
	{
		for (int y=0; y<gaussIterations.y; ++y)
		{
			cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
				Vec<size_t>(1,sizeconstKernelDims.y,1),	sizeconstKernelDims.x);
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();
		}
	}

	if (curChunk->getFullChunkSize().z>1)
	{
		for (int z=0; z<gaussIterations.z; ++z)
		{
			cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
				Vec<size_t>(1,1,sizeconstKernelDims.z),	sizeconstKernelDims.y+sizeconstKernelDims.x);
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();
		}
	}
}

template <class PixelType>
PixelType* cGaussianFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas, PixelType** imageOut=NULL, int device=0)
{
    cudaSetDevice(device);
	PixelType* imOut = setUpOutIm(dims, imageOut);

	Vec<int> gaussIterations(0,0,0);
	sigmas.x = (dims.x==1) ? (0) : (sigmas.x);
	sigmas.y = (dims.y==1) ? (0) : (sigmas.y);
	sigmas.z = (dims.z==1) ? (0) : (sigmas.z);

	float* hostKernel;

	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,&hostKernel,gaussIterations);
	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

    int blockSize = getKernelMaxThreads(cudaMultAddFilter<PixelType>);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,sizeconstKernelDims,blockSize);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());

		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());

		runGaussIterations(gaussIterations, curChunk, deviceImages, sizeconstKernelDims,device);

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	delete[] hostKernel;

	return imOut;
}
