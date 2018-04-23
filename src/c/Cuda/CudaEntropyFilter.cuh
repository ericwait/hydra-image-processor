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
#include "CudaGetMinMax.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <omp.h>

template <class PixelTypeIn>
__global__ void cudaEntropyFilter(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<float> imageOut, Kernel constKernelMem, const float minValue, const float maxValue)
{
	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		unsigned int histogram[256];
		double binWidth = (double)(maxValue - minValue) / 256.0;
		for (int i = 0; i < 255; ++i)
			histogram[i] = 0;

		KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDims());
		double outVal = 0;
		unsigned int count = 0;
		for (; !kIt.end(); ++kIt)
		{
			Vec<float> imInPos = kIt.getImageCoordinate();
			double inVal = (double)imageIn(imInPos);
			Vec<size_t> coord = kIt.getKernelCoordinate();
			float kernVal = constKernelMem(coord);

			if (kernVal != 0.0f)
			{
				int binNum = floor((double)(inVal * kernVal - minValue) / binWidth);
				++(histogram[binNum]);
				++count;
			}
		}

		for (int i=0; i<255; ++i)
		{
			double val = (double)(histogram[i]) / (double)count;
			if (val > 0)
				outVal += val*log2(val);
		}

		imageOut(threadCoordinate) = (float)-outVal;
	}
}


template <class PixelTypeIn>
void cEntropyFilter(ImageContainer<PixelTypeIn> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int device = -1)
{
	float minVal, maxVal;
	cGetMinMax(imageIn.getPtr(), imageIn.getNumElements(), minVal, maxVal, device);
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm<float>(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(cudaEntropyFilter<PixelTypeIn>, device);

	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, sizeof(float), kernel.getSpatialDims());

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<float> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		Kernel constKernelMem(kernel, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			cudaEntropyFilter<<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem, MIN_VAL, MAX_VAL);
			
			chunks[i].retriveROI(imageOut, deviceImages.getNextBuffer());
		}

		constKernelMem.clean();
	}
}
