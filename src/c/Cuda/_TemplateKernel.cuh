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
#include <omp.h>

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaFooFilter(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut,
	Kernel constKernelMem, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getSpatialDims())
	{
		KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDimensions());
		for (; !kIt.lastFrame(); kIt.nextFrame())
		{
			for (; !kIt.lastChannel(); kIt.nextChannel())
			{
				double outVal = 0;
				for (; !kIt.lastPosition(); ++kIt)
				{
					Vec<float> imInPos = kIt.getImageCoordinate();
					double inVal = (double)imageIn(imInPos, kIt.getChannel(), kIt.getFrame());
					float kernVal = constKernelMem(kIt.getKernelCoordinate());

					//////////////////////////////////////////////////////////
					// Do something interesting here
					// e.g. convolution
					// outVal += inVal * kernval;
					//////////////////////////////////////////////////////////
				}
				ImageDimensions outPos = ImageDimensions(threadCoordinate, kIt.getChannel(), kIt.getFrame());
				imageOut(outPos) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
			}
		}
	}
}


template <class PixelTypeIn, class PixelTypeOut>
void cFooFilter(ImageContainer<PixelTypeIn> imageIn, ImageContainer<PixelTypeOut>& imageOut,
	ImageContainer<float> kernel, int numIterations = 1, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm<PixelTypeOut>(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(cudaMaxFilter<PixelTypeIn, PixelTypeOut>, device);

	size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize,
		kernel.getSpatialDims());

	ImageDimensions maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_dynamic(0);
	omp_set_num_threads(2);
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);
		Kernel constKernelMem(kernel, CUR_DEVICE);

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			chunks[i].sendROI(imageIn, deviceImages.getCurBuffer());
			deviceImages.setAllDims(chunks[i].getFullChunkSize());
			DEBUG_KERNEL_CHECK();

			for (int j = 0; j < numIterations; ++j)
			{
				cudaFooFilter << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()),
					*(deviceImages.getNextBuffer()), constKernelMem, MIN_VAL, MAX_VAL);
				DEBUG_KERNEL_CHECK();
				deviceImages.incrementBuffer();
			}
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}

		constKernelMem.clean();
	}
}
