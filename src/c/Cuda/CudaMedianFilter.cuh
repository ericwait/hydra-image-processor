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

template <class PixelType>
__device__ PixelType* SubDivide(PixelType* pB, PixelType* pE)
{
	PixelType* pPivot = --pE;
	const PixelType pivot = *pPivot;

	while (pB < pE)
	{
		if (*pB > pivot)
		{
			--pE;
			PixelType temp = *pB;
			*pB = *pE;
			*pE = temp;
		}
		else
			++pB;
	}

	PixelType temp = *pPivot;
	*pPivot = *pE;
	*pE = temp;

	return pE;
}

template <class PixelType>
__device__ void SelectElement(PixelType* pB, PixelType* pE, size_t k)
{
	while (true)
	{
		PixelType* pPivot = SubDivide(pB, pE);
		size_t n = pPivot - pB;

		if (n == k)
			break;

		if (n > k)
			pE = pPivot;
		else
		{
			pB = pPivot + 1;
			k -= (n + 1);
		}
	}
}

template <class PixelType>
__device__ PixelType cudaFindMedian(PixelType* vals, int numVals)
{
	SelectElement(vals, vals + numVals, numVals / 2);
	return vals[numVals / 2];
}

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaMedianFilter(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, Kernel constKernelMem, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	extern __shared__ unsigned char valsShared[];
	PixelTypeIn* vals = (PixelTypeIn*)valsShared;

	Vec<size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		Vec<size_t> blockDimVec(blockDim.x, blockDim.y, blockDim.z);
		size_t linearThreadIdx = blockDimVec.linearAddressAt(Vec<size_t>(threadIdx.x, threadIdx.y, threadIdx.z));
		int sharedMemOffset = linearThreadIdx*constKernelMem.getDims().product();
		int kernelVolume = 0;

		KernelIterator kIt(threadCoordinate, imageIn.getDims(), constKernelMem.getDims());
		for (; !kIt.end(); ++kIt)
		{
			Vec<float> imInPos = kIt.getImageCoordinate();
			double inVal = (double)imageIn(imInPos);
			Vec<size_t> coord = kIt.getKernelCoordinate();
			float kernVal = constKernelMem(coord);

			if (kernVal != 0.0f)
			{
				vals[kernelVolume + sharedMemOffset] = inVal * kernVal;
				++kernelVolume;
			}
		}
		PixelTypeOut outVal = (PixelTypeOut)cudaFindMedian(vals + sharedMemOffset, kernelVolume);
		imageOut(threadCoordinate) = CLAMP(outVal, minValue, maxValue);
	}
}


template <class PixelTypeIn, class PixelTypeOut>
void cMedianFilter(ImageContainer<PixelTypeIn> imageIn, ImageContainer<PixelTypeOut>& imageOut, ImageContainer<float> kernel, int numIterations = 1, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	setUpOutIm<PixelTypeOut>(imageIn.getDims(), imageOut);

	if (kernel.getSpatialDims()==Vec<size_t>(1))
	{
		if (std::is_same<PixelTypeIn, PixelTypeOut>::value)
		{
			memcpy(imageOut.getPtr(), imageIn.getPtr(), sizeof(PixelTypeOut)*imageOut.getNumElements());
		}
		else
		{
			PixelTypeOut* outPtr = imageOut.getPtr();
			PixelTypeIn* inPtr = imageIn.getPtr();
			for (size_t i = 0; i < imageOut.getNumElements(); ++i)
			{
				outPtr[i] = (PixelTypeOut)inPtr[i];
			}
		}
		return;
	}

	CudaDevices cudaDevs(cudaMedianFilter<PixelTypeIn, PixelTypeOut>, device);

	size_t sizeOfsharedMem = kernel.getNumElements() * sizeof(PixelTypeIn);
	size_t numThreads = (size_t)floor((double)cudaDevs.getMinSharedMem() / (double)sizeOfsharedMem);
	
	if (numThreads < 32) // TODO: Use global memory
		throw std::runtime_error("Median neighborhood is too large to fit in shared memory on the GPU");

	numThreads = MIN(numThreads, cudaDevs.getMaxThreadsPerBlock());
	cudaDevs.setMaxThreadsPerBlock(numThreads);

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
		Kernel constKernelMem(kernel, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			size_t sharedMemorysize = kernel.getNumElements() * sizeof(PixelTypeIn) * chunks[i].threads.x * chunks[i].threads.y * chunks[i].threads.z;

			for (int j = 0; j < numIterations; ++j)
			{
				cudaMedianFilter<<<chunks[i].blocks, chunks[i].threads,sharedMemorysize>>>(*(deviceImages.getCurBuffer()),
					*(deviceImages.getNextBuffer()), constKernelMem, MIN_VAL, MAX_VAL);
				deviceImages.incrementBuffer();
			}
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}

		constKernelMem.clean();
	}
}
