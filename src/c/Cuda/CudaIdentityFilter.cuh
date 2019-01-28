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
__global__ void cudaIdentity(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut)
{
	Vec<std::size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if ( threadCoordinate < imageIn.getDims() )
	{
		imageOut(threadCoordinate) = (PixelTypeOut)imageIn(threadCoordinate);
	}
}


template <class PixelTypeIn, class PixelTypeOut>
void cIdentityFilter(ImageView<PixelTypeIn> imageIn, ImageView<PixelTypeOut> imageOut, int device = -1)
{
	const int NUM_BUFF_NEEDED = 2;

	CudaDevices cudaDevs(cudaIdentity<PixelTypeIn, PixelTypeOut>, device);

	Vec<std::size_t> kernDims(5*75+1);

	std::size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, kernDims);

	Vec<std::size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);

		for ( int i = CUDA_IDX; i < chunks.size(); i += N_THREADS )
		{
			if ( !chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()) )
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			cudaIdentity<<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()));
			deviceImages.incrementBuffer();

			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}
	}
}
