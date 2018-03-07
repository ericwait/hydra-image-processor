#pragma once
#include "CudaImageContainer.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaUtilities.h"
#include "CudaDeviceInfo.h"
#include "Kernel.cuh"
#include "KernelIterator.cuh"
#include "ImageDimensions.cuh"
#include "ImageChunk.h"
#include "Defines.h"
#include "Vec.h"

#include <cuda_runtime.h>
#include <limits>


template <class PixelTypeIn, class PixelTypeOut>
__global__ void maxFilter(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, Kernel constKernelMem, PixelTypeOut minValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getSpatialDims())
	{
		KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDimensions());
		for (; !kIt.end(); kIt.nextFrame())
		{
			for (; !kIt.frameEnd(); kIt.nextChannel())
			{
				PixelTypeOut outVal = minValue;
				for (; !kIt.channelEnd(); ++kIt)
				{
					Vec<float> imInPos = kIt.getImageCoordinate();
					PixelTypeIn inVal = imageIn(imInPos, kIt.getChannel(), kIt.getFrame());
					float kernVal = constKernelMem[kIt.getKernelIdx()];

					if (kernVal != 0.0f)
						if (outVal < inVal)
							outVal = inVal;
				}
				ImageDimensions outPos = ImageDimensions(threadCoordinate,kIt.getChannel(), kIt.getFrame());

				imageOut(outPos) = outVal;
			}
		}
	}
}

template <class PixelTypeIn, class PixelTypeOut>
void cMaxFilter(const ImageContainer<PixelTypeIn> imageIn, ImageContainer<PixelTypeOut>& imageOut, Vec<size_t> kernelDims, float* kernel = NULL, int numIterations=1, int device = -1)
{
	const PixelTypeOut minVal = std::numeric_limits<PixelTypeOut>::lowest();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm<PixelTypeOut>(imageIn.getDims(), imageOut);
	Kernel constKernelMem(kernelDims, kernel);

	CudaDevices cudaDevs(maxFilter<PixelTypeIn, PixelTypeOut>, device);

	size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, kernelDims);

	ImageDimensions maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	{// This is were the openMP should go
		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED,maxDeviceDims, cudaDevs.getDeviceIdx(0));

		for (std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk != chunks.end(); ++curChunk)
		{
			curChunk->sendROI(imageIn, deviceImages.getCurBuffer());
			deviceImages.setNextDims(curChunk->getFullChunkSize());

			for (int i = 0; i < numIterations; ++i)
			{
				maxFilter<<<curChunk->blocks, curChunk->threads>>> (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()),constKernelMem, minVal);
				DEBUG_KERNEL_CHECK();
			}

			deviceImages.incrementBuffer();
			curChunk->retriveROI(imageOut, deviceImages.getCurBuffer());
		}
	}

	constKernelMem.clean();
}
