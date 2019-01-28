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

template <class PixelTypeIn1, class PixelTypeIn2, class PixelTypeOut>
__global__ void cudaAddTwoImages(CudaImageContainer<PixelTypeIn1> imageIn1, CudaImageContainer<PixelTypeIn2> imageIn2, CudaImageContainer<PixelTypeOut> imageOut, PixelTypeOut minValue, PixelTypeOut maxValue, double image2Factor=1.0)
{
	Vec<std::size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn1.getDims() && threadCoordinate<imageIn2.getDims())
	{
		double val = imageIn2(threadCoordinate)*image2Factor;
		val += imageIn1(threadCoordinate);

		imageOut(threadCoordinate) = (PixelTypeOut)CLAMP(val, minValue, maxValue);
	}
}


template <class PixelTypeIn1, class PixelTypeIn2, class PixelTypeOut>
void cAddTwoImages(ImageView<PixelTypeIn1> imageIn1, ImageView<PixelTypeIn2> imageIn2, ImageView<PixelTypeOut> imageOut, double image2Factor=1.0, int device=-1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 3;

	CudaDevices cudaDevs(cudaAddTwoImages<PixelTypeIn1, PixelTypeIn2, PixelTypeOut>, device);

	std::size_t maxTypeSize = MAX(sizeof(PixelTypeIn1), MAX(sizeof(PixelTypeIn1), sizeof(PixelTypeOut)));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn1.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize);

	Vec<std::size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeIn1> deviceIn1(1, maxDeviceDims, CUR_DEVICE);
		CudaDeviceImages<PixelTypeIn2> deviceIn2(1, maxDeviceDims, CUR_DEVICE);
		CudaDeviceImages<PixelTypeOut> deviceOut(1, maxDeviceDims, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn1, deviceIn1.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			if (!chunks[i].sendROI(imageIn2, deviceIn2.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceIn1.setAllDims(chunks[i].getFullChunkSize());
			deviceIn2.setAllDims(chunks[i].getFullChunkSize());
			deviceOut.setAllDims(chunks[i].getFullChunkSize());

			cudaAddTwoImages<<<chunks[i].blocks, chunks[i].threads>>>(*(deviceIn1.getCurBuffer()), *(deviceIn2.getCurBuffer()), *(deviceOut.getCurBuffer()), image2Factor, MIN_VAL, MAX_VAL);
			chunks[i].retriveROI(imageOut, deviceOut.getCurBuffer());
		}
	}
}
