#pragma once
#include "CudaThreshold.cuh"
#include "CudaMaxFilter.cuh"
#include "CudaMinFilter.cuh"

template <class PixelType>
PixelType* cSegment(const PixelType* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel=NULL,
					PixelType** imageOut=NULL, int device=0)
{
    cudaSetDevice(device);
	PixelType thresh = cOtsuThresholdValue(imageIn,dims,device);
	thresh = (PixelType)(thresh*alpha);

	PixelType* imOut = setUpOutIm(dims, imageOut);

	PixelType minVal = std::numeric_limits<PixelType>::min();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	if (kernel==NULL)
	{
		kernelDims = kernelDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernelDims.product()];
		for (int i=0; i<kernelDims.product(); ++i)
			ones[i] = 1.0f;

		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(float)*kernelDims.product()));
		delete[] ones;
	} 
	else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*kernelDims.product()));
	}

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,kernelDims);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);
	DEBUG_KERNEL_CHECK();

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

 		cudaThreshold<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),thresh,minVal,maxVal);
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();

		cudaMinFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernelDims,
			minVal,maxVal);
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();

		cudaMaxFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernelDims,
			minVal,maxVal);
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}