#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "CHelpers.h"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"

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
		} else
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
	SelectElement(vals,vals+numVals, numVals/2);
	return vals[numVals/2];
}

template <class PixelType>
__global__ void cudaMedianFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
								 Vec<size_t> hostKernelDims )
{
	extern __shared__ DevicePixelType vals[];
	DeviceVec<size_t> kernelDims = hostKernelDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  kernelDims.product();

	if (coordinate<imageIn.getDeviceDims())
	{
		int kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> halfKernal = kernelDims/2;

		DeviceVec<size_t> curCoordIm = coordinate - halfKernal;
		curCoordIm.z = (coordinate.z<halfKernal.z) ? 0 : coordinate.z-halfKernal.z;
		for (; curCoordIm.z<=coordinate.z+halfKernal.z && curCoordIm.z<imageIn.getDeviceDims().z; ++curCoordIm.z)
		{
			curCoordIm.y = (coordinate.y<halfKernal.y) ? 0 : coordinate.y-halfKernal.y/2;
			for (; curCoordIm.y<=coordinate.y+halfKernal.y && curCoordIm.y<imageIn.getDeviceDims().y; ++curCoordIm.y)
			{
				curCoordIm.x = (coordinate.x<halfKernal.x) ? 0 : coordinate.x-halfKernal.x/2;
				for (; curCoordIm.x<=coordinate.x+halfKernal.x && curCoordIm.x<imageIn.getDeviceDims().x; ++curCoordIm.x)
				{
					vals[kernelVolume+offset] = imageIn[curCoordIm];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinate] = cudaFindMedian(vals+offset,kernelVolume);
	}
	__syncthreads();
}

template <class PixelType>
void runMedianFilter(cudaDeviceProp& props, std::vector<ImageChunk>::iterator curChunk, Vec<size_t> &neighborhood, 
					 CudaDeviceImages<PixelType>& deviceImages)
{
	dim3 blocks(curChunk->blocks);
	dim3 threads(curChunk->threads);
	double threadVolume = threads.x * threads.y * threads.z;
	double newThreadVolume = (double)props.sharedMemPerBlock/(sizeof(PixelType)*neighborhood.product());

	if (newThreadVolume<threadVolume)
	{
		double alpha = pow(threadVolume/newThreadVolume,1.0/3.0);
		threads.x = (unsigned int)(threads.x / alpha);
		threads.y = (unsigned int)(threads.y / alpha);
		threads.z = (unsigned int)(threads.z / alpha);
		threads.x = (threads.x>0) ? (threads.x) : (1);
		threads.y = (threads.y>0) ? (threads.y) : (1);
		threads.z = (threads.z>0) ? (threads.z) : (1);

		blocks.x = (unsigned int)ceil((double)curChunk->getFullChunkSize().x / threads.x);
		blocks.y = (unsigned int)ceil((double)curChunk->getFullChunkSize().y / threads.y);
		blocks.z = (unsigned int)ceil((double)curChunk->getFullChunkSize().z / threads.z);
	}

	size_t sharedMemorysize = neighborhood.product()*sizeof(PixelType) * threads.x * threads.y * threads.z;

	cudaMedianFilter<<<blocks,threads,sharedMemorysize>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),neighborhood);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();
}

template <class PixelType>
PixelType* medianFilter(const PixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, PixelType** imageOut=NULL, int device=0)
{
	PixelType* imOut = setUpOutIm(dims, imageOut);

	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t memAvail, total;
	cudaMemGetInfo(&memAvail,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(memAvail*MAX_MEM_AVAIL),props,neighborhood);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		runMedianFilter(props, curChunk, neighborhood, deviceImages);

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}