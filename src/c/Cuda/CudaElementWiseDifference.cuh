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

template <class PixelType1In, class PixelType2In, class PixelTypeOut>
__global__ void cudaElementWiseDifference(CudaImageContainer<PixelType1In> image1In, CudaImageContainer<PixelType2In> image2In, CudaImageContainer<PixelTypeOut> imageOut, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	Vec<size_t> minDims = Vec<size_t>::min(image1In.getDims(), image2In.getDims());
	if (threadCoordinate<minDims)
	{
		double outVal = ((double)(image1In(threadCoordinate)) - (double)(image2In(threadCoordinate)));
		imageOut(threadCoordinate) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
	}
}


template <class PixelType1In, class PixelType2In, class PixelTypeOut>
void cElementWiseDifference(ImageContainer<PixelType1In> image1In, ImageContainer<PixelType2In> image2In, ImageContainer<PixelTypeOut>& imageOut, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 3;

	ImageDimensions maxDims(Vec<size_t>::max(image1In.getSpatialDims(), image2In.getSpatialDims()),MAX(image1In.getNumChannels(),image2In.getNumChannels()), MAX(image1In.getNumFrames(),image2In.getNumFrames()));
	setUpOutIm<PixelTypeOut>(maxDims, imageOut);

	CudaDevices cudaDevs(cudaElementWiseDifference<PixelType1In, PixelType2In, PixelTypeOut>, device);

	size_t maxTypeSize = MAX(MAX(sizeof(PixelType1In),sizeof(PixelType2In)), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(maxDims, NUM_BUFF_NEEDED, cudaDevs, maxTypeSize);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);
		HANDLE_ERROR(cudaSetDevice(CUR_DEVICE)); // This is done because there is no Kernel type to set it for us

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(image1In, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");
			if (!chunks[i].sendROI(image2In, deviceImages.getNextBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());
			DEBUG_KERNEL_CHECK();

			cudaElementWiseDifference << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()),				*(deviceImages.getNextBuffer()), *(deviceImages.getThirdBuffer()), MIN_VAL, MAX_VAL);
			DEBUG_KERNEL_CHECK();

			chunks[i].retriveROI(imageOut, deviceImages.getThirdBuffer());
		}
	}
}
