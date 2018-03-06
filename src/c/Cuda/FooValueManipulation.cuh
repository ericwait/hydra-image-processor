#pragma once
#include "CudaImageContainer.cuh"
#include "CudaUtilities.cuh"
#include "KernelIterator.cuh"
#include "ImageDimensions.cuh"
#include "Defines.h"
#include <cuda_runtime.h>

template <class PixelTypeIn, class PixelTypeOut>
__global__ void fooSingleOperation(CudaImageContainer<PixelTypeIn> imageIn,
	CudaImageContainer<PixelTypeOut> imageOut,
	PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> coordinate = GetThreadBlockCoordinate();

	if (coordinate < imageIn.getDims())
	{
		Vec<size_t> neighboorhood = Vec<size_t>(1);
		KernelIterator kIt(coordinate, imageIn.getDims(), neighboorhood);
		for (; !kIt.end(); ++kIt)
		{
			ImageDimensions pos = kIt.getFullPos();
			PixelTypeIn val = imageIn(pos);
			// do something interesting here
			imageOut(pos) = CLAMP(val, minValue, maxValue);
		}
	}
}

template <class PixelTypeIn, class PixelTypeOut>
void cFooSingleOperation(const ImageContainer<PixelTypeIn> imageIn,
	ImageContainer<PixelTypeOut>& imageOut, int device = -1)
{
	const PixelTypeOut minVal = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut maxVal = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(fooSingleOperation<PixelTypeIn,PixelTypeOut>, device);

	size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	Vec<size_t> neighboorhood = Vec<size_t>(1);
	std::vector<ImageChunk> chunks = 
		calculateBuffers<PixelTypeOut>(imageIn.getDims(), NUM_BUFF_NEEDED,
			cudaDevs, maxTypeSize, neighboorhood);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	{// This is were the openMP should go
		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED,
			maxDeviceDims, deviceIdxList[0]);

		for (std::vector<ImageChunk>::iterator curChunk = chunks.begin();
			curChunk != chunks.end(); ++curChunk)
		{
			curChunk->sendROI(imageIn, deviceImages.getCurBuffer());
			deviceImages.setNextDims(curChunk->getFullChunkSize());

			fooSingleOperation<<<curChunk->blocks, curChunk->threads>>>
				(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()),
					minVal, maxVal);
			DEBUG_KERNEL_CHECK();

			deviceImages.incrementBuffer();
			curChunk->retriveROI(imageOut, deviceImages.getCurBuffer());
		}
	}
}
