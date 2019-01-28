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

#include "CudaMeanAndVariance.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <omp.h>

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaWienerFilter(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, Kernel constKernelMem, double noiseVariance, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<std::size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		double mu = 0.0, var = 0.0;
		deviceMeanAndVariance(threadCoordinate, imageIn, constKernelMem, mu, var);

		double factor = MAX(0.0, var - noiseVariance) / var;

		double outVal = mu + factor * (imageIn(threadCoordinate) - mu);

		imageOut(threadCoordinate) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
	}
}

template <class PixelTypeIn, class PixelTypeOut>
void cWienerFilter(ImageView<PixelTypeIn> imageIn, ImageView<PixelTypeOut> imageOut, ImageView<float> kernel, double noiseVariance= -1.0, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	CudaDevices cudaDevs(cudaWienerFilter<PixelTypeIn, PixelTypeOut>, device);

	std::size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, kernel.getSpatialDims());

	Vec<std::size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		Kernel constKernelMem(kernel, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			double lclNoiceVar = noiseVariance;
			if (lclNoiceVar == -1.0)
			{
				cudaVarFilter<<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem, MIN_VAL, MAX_VAL);
				double sum = 0.0;
				sumBuffer(chunks[i], deviceImages.getNextBuffer(), sum);
				lclNoiceVar = sum / chunks[i].getFullChunkSize().product();// this differs from MATLAB because of the edge errors that MATLAB has in its neighborhood variance 
				PixelTypeIn minVal, maxVal;
				minMaxBuffer(chunks[i], deviceImages.getCurBuffer(), minVal, maxVal, cudaDevs.getMinSharedMem());
				lclNoiceVar *= SQR((double)maxVal);
			}

			cudaWienerFilter<<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem, lclNoiceVar, MIN_VAL, MAX_VAL);
			deviceImages.incrementBuffer();
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}

		constKernelMem.clean();
	}
}
