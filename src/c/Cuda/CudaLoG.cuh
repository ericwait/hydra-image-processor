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
#include "CudaAddTwoImages.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <omp.h>

template <class PixelTypeIn>
void cLoG(ImageContainer<PixelTypeIn> imageIn, ImageContainer<float>& imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1)
{
	const float MIN_VAL = std::numeric_limits<float>::lowest();
	const float MAX_VAL = std::numeric_limits<float>::max();
	const int NUM_BUFF_NEEDED = 3;

	setUpOutIm<float>(imageIn.getDims(), imageOut);

	CudaDevices cudaDevs(cudaAddTwoImages<float,float,float>, device);

	Vec<size_t> kernelDims(0);
	float* hostLoG_GausKernels = createLoG_GausKernels(sigmas, kernelDims);

	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, sizeof(float), kernelDims);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<float> deviceImages(NUM_BUFF_NEEDED-1, maxDeviceDims, CUR_DEVICE);
		CudaDeviceImages<float> deviceImagesScratch(1, maxDeviceDims, CUR_DEVICE);

		size_t logStride = kernelDims.sum();
		Kernel constFullKern(logStride*2, hostLoG_GausKernels, CUR_DEVICE);

		Kernel constLoGKernelMem_x = constFullKern.getOffsetCopy(Vec<size_t>(kernelDims.x, 1, 1), 0);
		Kernel constLoGKernelMem_y = constFullKern.getOffsetCopy(Vec<size_t>(1, kernelDims.y, 1), kernelDims.x);
		Kernel constLoGKernelMem_z = constFullKern.getOffsetCopy(Vec<size_t>(1, 1, kernelDims.z), kernelDims.x + kernelDims.y);

		Kernel constGausKernelMem_x = constFullKern.getOffsetCopy(Vec<size_t>(kernelDims.x, 1, 1), logStride);
		Kernel constGausKernelMem_y = constFullKern.getOffsetCopy(Vec<size_t>(1, kernelDims.y, 1), kernelDims.x + logStride);
		Kernel constGausKernelMem_z = constFullKern.getOffsetCopy(Vec<size_t>(1, 1, kernelDims.z), kernelDims.x + kernelDims.y + logStride);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			size_t memsize = sizeof(float)*chunks[i].getFullChunkSize().product();
			deviceImages.setAllDims(chunks[i].getFullChunkSize());
			deviceImagesScratch.setAllDims(chunks[i].getFullChunkSize());

			HANDLE_ERROR(cudaMemset(deviceImagesScratch.getCurBuffer()->getDeviceImagePointer(), 0, memsize));

			// apply LoG in X
			if (sigmas.x!=0)
			{
				if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
					std::runtime_error("Error sending ROI to device!");
				cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constLoGKernelMem_x, MIN_VAL, MAX_VAL, false);
				deviceImages.incrementBuffer();
				if (sigmas.y!=0)
				{
					cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constGausKernelMem_y, MIN_VAL, MAX_VAL);
					deviceImages.incrementBuffer();
				}
				if (sigmas.z!=0)
				{
					cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constGausKernelMem_z, MIN_VAL, MAX_VAL);
					deviceImages.incrementBuffer();
				}
				cudaAddTwoImages << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImagesScratch.getCurBuffer()), *(deviceImagesScratch.getCurBuffer()), MIN_VAL, MAX_VAL);
				DEBUG_KERNEL_CHECK();
			}

			// apply LoG in Y
			if (sigmas.y!=0)
			{
				if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
					std::runtime_error("Error sending ROI to device!");
				if (sigmas.x!=0)
				{
					cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constGausKernelMem_x, MIN_VAL, MAX_VAL);
					deviceImages.incrementBuffer();
				}
				cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constLoGKernelMem_y, MIN_VAL, MAX_VAL, false);
				deviceImages.incrementBuffer();
				if (sigmas.z!=0)
				{
					cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constGausKernelMem_z, MIN_VAL, MAX_VAL);
					deviceImages.incrementBuffer();
				}
				cudaAddTwoImages << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImagesScratch.getCurBuffer()), *(deviceImagesScratch.getCurBuffer()), MIN_VAL, MAX_VAL);
				DEBUG_KERNEL_CHECK();
			}

			// apply LoG in Z
			if (sigmas.z!=0)
			{
				if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
					std::runtime_error("Error sending ROI to device!");
				if (sigmas.x!=0)
				{
					cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constGausKernelMem_x, MIN_VAL, MAX_VAL);
					deviceImages.incrementBuffer();
				}
				if(sigmas.y!=0)
				{
					cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constGausKernelMem_y, MIN_VAL, MAX_VAL);
					deviceImages.incrementBuffer();
				}
				cudaMultiplySum << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constLoGKernelMem_z, MIN_VAL, MAX_VAL, false);
				deviceImages.incrementBuffer();
				cudaAddTwoImages << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImagesScratch.getCurBuffer()), *(deviceImagesScratch.getCurBuffer()), MIN_VAL, MAX_VAL);
				DEBUG_KERNEL_CHECK();
			}

			chunks[i].retriveROI(imageOut, deviceImagesScratch.getCurBuffer());
		}

		constFullKern.clean();
	}

	delete[] hostLoG_GausKernels;
}
