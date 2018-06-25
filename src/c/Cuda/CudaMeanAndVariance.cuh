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

template <class PixelTypeIn>
__device__ void deviceMean(Vec<size_t> threadCoordinate, CudaImageContainer<PixelTypeIn> &imageIn, Kernel &constKernelMem, double& mu)
{
	KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDims());
	mu = 0;
	size_t count = 0;
	for (; !kIt.end(); ++kIt)
	{
		Vec<float> imInPos = kIt.getImageCoordinate();
		double inVal = (double)imageIn(imInPos);
		Vec<size_t> coord = kIt.getKernelCoordinate();
		float kernVal = constKernelMem(coord);

		if (kernVal != 0.0f)
		{
			mu += inVal * kernVal;
			++count;
		}
	}
	mu /= (double)count;
}

template <class PixelTypeIn>
__device__ void deviceVariance(Vec<size_t> threadCoordinate, CudaImageContainer<PixelTypeIn> &imageIn, Kernel &constKernelMem, double& varOut, double mu)
{
	KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDims());
	varOut = 0;
	size_t count = 0;
	for (; !kIt.end(); ++kIt)
	{
		Vec<float> imInPos = kIt.getImageCoordinate();
		double inVal = (double)imageIn(imInPos);
		Vec<size_t> coord = kIt.getKernelCoordinate();
		float kernVal = constKernelMem(coord);

		if (kernVal != 0.0f)
		{
			varOut += SQR(kernVal*inVal) - SQR(mu);
			++count;
		}
	}

	varOut /= (double)count;
}

template <class PixelTypeIn>
__device__ void deviceMeanAndVariance(Vec<size_t> threadCoordinate, CudaImageContainer<PixelTypeIn> &imageIn, Kernel &constKernelMem, double& mu, double& var)
{
	mu = 0.0;
	deviceMean(threadCoordinate, imageIn, constKernelMem, mu);

	var = 0.0;
	deviceVariance(threadCoordinate, imageIn, constKernelMem, var, mu);
}

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaMeanAndVariance(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> muOut, CudaImageContainer<PixelTypeOut> varOut, Kernel constKernelMem, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		double mu = 0.0, var = 0.0;
		deviceMeanAndVariance(threadCoordinate, imageIn, constKernelMem, mu, var);

		muOut(threadCoordinate) = (PixelTypeOut)CLAMP(mu, minValue, maxValue);
		varOut(threadCoordinate) = (PixelTypeOut)CLAMP(var, minValue, maxValue);
	}
}

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaMeanAndStd(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> muOut, CudaImageContainer<PixelTypeOut> varOut, Kernel constKernelMem, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		double mu = 0.0, var = 0.0;
		deviceMeanAndVariance(threadCoordinate, imageIn, constKernelMem, mu, var);

		muOut(threadCoordinate) = (PixelTypeOut)CLAMP(mu, minValue, maxValue);
		varOut(threadCoordinate) = (PixelTypeOut)CLAMP(sqrt(var), minValue, maxValue);
	}
}

template <class PixelTypeIn, class PixelTypeOut>
void cMeanAndVariance(ImageContainer<PixelTypeIn> imageIn, ImageContainer<PixelTypeOut>& muOut, ImageContainer<PixelTypeOut>& varOut, ImageContainer<float> kernel, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 3;

	setUpOutIm<PixelTypeOut>(imageIn.getDims(), muOut);
	setUpOutIm<PixelTypeOut>(imageIn.getDims(), varOut);

	CudaDevices cudaDevs(cudaMeanAndVariance<PixelTypeIn, PixelTypeOut>, device);

	size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, kernel.getSpatialDims());

	Vec<size_t> maxDeviceDims;
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

			cudaMeanAndVariance<<<chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), *(deviceImages.getThirdBuffer()), constKernelMem, MIN_VAL, MAX_VAL);

			chunks[i].retriveROI(muOut, deviceImages.getNextBuffer());
			chunks[i].retriveROI(varOut, deviceImages.getThirdBuffer());
		}

		constKernelMem.clean();
	}
}

