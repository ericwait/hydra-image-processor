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
__global__ void cudaMinFilter(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, Kernel constKernelMem, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDims());
		double outVal = 0;
		for (; !kIt.end(); ++kIt)
		{
			Vec<float> imInPos = kIt.getImageCoordinate();
			double inVal = (double)imageIn(imInPos);
			Vec<size_t> coord = kIt.getKernelCoordinate();
			float kernVal = constKernelMem(coord);

			if (kernVal != 0.0f)
			{
				inVal = inVal*kernVal;
				if (outVal > inVal)
					outVal = inVal;
			}
		}
		imageOut(threadCoordinate) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
	}
}


template <class PixelTypeIn, class PixelTypeOut>
void cMinFilter(ImageContainer<PixelTypeIn> imageIn, ImageContainer<PixelTypeOut>& imageOut, ImageContainer<float> kernel, int numIterations = 1, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm<PixelTypeOut>(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(cudaMinFilter<PixelTypeIn, PixelTypeOut>, device);

	size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize,kernel.getSpatialDims());

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		Kernel constKernelMem(kernel,CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());
			DEBUG_KERNEL_CHECK();

			for (int j = 0; j < numIterations; ++j)
			{
				cudaMinFilter << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()), constKernelMem, MIN_VAL, MAX_VAL);
				DEBUG_KERNEL_CHECK();
				deviceImages.incrementBuffer();
			}
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}

		constKernelMem.clean();
	}
}
