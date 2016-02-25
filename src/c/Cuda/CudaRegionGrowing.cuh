#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"
#include "Vec.h"
#include <vector>
#include "CHelpers.h"
#include "CudaUtilities.cuh"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template<class PixelType>
__global__ void cudaRegionGrowing(CudaImageContainer<PixelType> imageIn,CudaImageContainer<bool> maskIn,CudaImageContainer<bool> maskOut,
	Vec<size_t> hostKernelDims,double threshold)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if(coordinate<imageIn.getDeviceDims())
	{
		if(maskIn[coordinate]==true)
		{
			maskOut[coordinate] = true;
		}
		else
		{
			PixelType curPixelVal = imageIn[coordinate] + threshold;
			DeviceVec<size_t> kernelDims = hostKernelDims;
			DeviceVec<size_t> halfKernal = kernelDims/2;

			DeviceVec<size_t> curCoordIm = coordinate - halfKernal;
			curCoordIm.z = (coordinate.z<halfKernal.z) ? 0 : coordinate.z-halfKernal.z;
			for(; curCoordIm.z<=coordinate.z+halfKernal.z && curCoordIm.z<imageIn.getDeviceDims().z; ++curCoordIm.z)
			{
				curCoordIm.y = (coordinate.y<halfKernal.y) ? 0 : coordinate.y-halfKernal.y;
				for(; curCoordIm.y<=coordinate.y+halfKernal.y && curCoordIm.y<imageIn.getDeviceDims().y; ++curCoordIm.y)
				{
					curCoordIm.x = (coordinate.x<halfKernal.x) ? 0 : coordinate.x-halfKernal.x;
					for(; curCoordIm.x<=coordinate.x+halfKernal.x && curCoordIm.x<imageIn.getDeviceDims().x; ++curCoordIm.x)
					{
						if(curPixelVal > imageIn[curCoordIm] && maskIn[curCoordIm]==true)
						{
							maskOut[coordinate] = true;
						}
					}
				}
			}
		}
	}
}

template <class PixelType>
void cRegionGrowing(const PixelType* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,int device=0)
{
	cudaSetDevice(device);
	PixelType minVal = std::numeric_limits<PixelType>::lowest();
	PixelType maxVal = std::numeric_limits<PixelType>::max();

	if(kernel==NULL)
	{
		kernelDims = kernelDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernelDims.product()];
		for(int i=0; i<kernelDims.product(); ++i)
			ones[i] = 1.0f;

		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,ones,sizeof(float)*kernelDims.product()));
		delete[] ones;
	} else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,kernel,sizeof(float)*kernelDims.product()));
	}

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem,total;
	cudaMemGetInfo(&availMem,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,kernelDims);

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks,maxDeviceDims);

	CudaDeviceImages<PixelType> deviceImages(1,maxDeviceDims,device);
	CudaDeviceImages<bool> deviceMask(2,maxDeviceDims,device);

	for(std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());
		curChunk->sendROI(imageMask,dims,deviceMask.getCurBuffer());
		deviceMask.setNextDims(curChunk->getFullChunkSize());

		cudaRegionGrowing<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceMask.getCurBuffer()),
			*(deviceMask.getNextBuffer()),kernelDims,threshold);
		DEBUG_KERNEL_CHECK();

		curChunk->retriveROI(imageMask,dims,deviceMask.getNextBuffer());
	}
}
