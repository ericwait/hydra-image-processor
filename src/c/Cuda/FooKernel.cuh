#pragma once
#include "CudaImageContainer.cuh"
#include "CudaUtilities.cuh"
#include "KernelIterator.cuh"
#include "ImageDimensions.cuh"
#include "Defines.h"
#include "Vec.h"

#include <cuda_runtime.h>

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelTypeIn, class PixelTypeOut>
__global__ void fooKernel(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, Vec<size_t> hostKernelDims, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> coordinate = GetThreadBlockCoordinate();

	if (coordinate < imageIn.getDims())
	{
		Vec<size_t> kernelDims = hostKernelDims;
		KernelIterator kIt(coordinate, imageIn.getDims(), kernelDims);
		for (; !kIt.end(); kIt.nextFrame())
		{
			for (; !kIt.frameEnd(); kIt.nextChannel())
			{
				PixelTypeOut outVal = 0;
				for (; !kIt.channelEnd(); ++kIt)
				{
					Vec<float> imInPos = kIt.getImageCoordinate();
					PixelTypeIn inVal = imageIn(imInPos,kIt.getChannel(),kIt.getFrame());
					float kernVal = cudaConstKernel[kIt.getKernelIdx()];
					// do something interesting here e.g. convolution:
					outVal += inVal * kernVal;
				}
				ImageDimensions outPos = ImageDimensions(coordinate, kIt.getChannel(), kIt.getFrame());
				imageOut(outPos) = outVal;
			}
		}
	}
}

template <class PixelTypeIn, class PixelTypeOut>
void cFooKernel(const ImageContainer<PixelTypeIn> imageIn, ImageContainer<PixelTypeOut>& imageOut, Vec<size_t> kernelDims, float* kernel=NULL, int device = -1)
{
	const PixelTypeOut minVal = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut maxVal = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(fooKernel<PixelTypeIn, PixelTypeOut>, device);

	size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers<PixelTypeOut>(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, kernelDims);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	{// This is were the openMP should go
		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, deviceIdxList[0]);

		for (std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk != chunks.end(); ++curChunk)
		{
			curChunk->sendROI(imageIn, deviceImages.getCurBuffer());
			deviceImages.setNextDims(curChunk->getFullChunkSize());

			fooSingleOperation << <curChunk->blocks, curChunk->threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), minVal, maxVal);
			DEBUG_KERNEL_CHECK();

			deviceImages.incrementBuffer();
			curChunk->retriveROI(imageOut, deviceImages.getCurBuffer());
		}
	}
}
