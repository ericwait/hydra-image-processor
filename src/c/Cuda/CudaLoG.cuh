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
#include "KernelGenerators.h"
#include "SeparableMultiplySum.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <omp.h>

template <class PixelTypeIn>
void cLoG(ImageContainer<PixelTypeIn> imageIn, ImageContainer<float>& imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1)
{
	const float MIN_VAL = std::numeric_limits<float>::lowest();
	const float MAX_VAL = std::numeric_limits<float>::max();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm<float>(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(cudaMultiplySum<PixelTypeIn, float>, device);

	Vec<size_t> kernDims(0);
	float* hostKernels = createLoGKernel(sigmas, kernDims);

	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, sizeof(float), kernDims);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	int numThreads = MIN(chunks.size(), cudaDevs.getNumDevices());
	omp_set_num_threads(numThreads);
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<float> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);

		Kernel constKernelMem(kernDims.product(), hostKernels, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			for (int j = 0; j < numIterations; ++j)
			{
				cudaMultiplySum<<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem, MIN_VAL, MAX_VAL);
				deviceImages.incrementBuffer();
			}
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}

	constKernelMem.clean();
	}

	delete[] hostKernels;
}
