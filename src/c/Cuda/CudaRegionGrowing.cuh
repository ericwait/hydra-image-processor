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

__device__ bool lineConnect(const CudaImageContainer<bool>& maskIn,DeviceVec<long int> prevCoord,DeviceVec<long int> nextCoord)
{
	if(prevCoord>=DeviceVec<long int>(0,0,0) && nextCoord>=DeviceVec<long int>(0,0,0))
	{
		if(prevCoord<maskIn.getDeviceDims() && nextCoord<maskIn.getDeviceDims())
		{

			if(maskIn[DeviceVec<size_t>(prevCoord)] && maskIn[DeviceVec<size_t>(nextCoord)])
				return true;
		}
	}
	
	return false;
}

__device__ bool willConnect(const CudaImageContainer<bool>& maskIn, DeviceVec<size_t> coordinateIn)
{
	if(maskIn[coordinateIn])
		return true;

	DeviceVec<long int> coordinate(coordinateIn);
	DeviceVec<long int> prevCoord;
	DeviceVec<long int> nextCoord;
	DeviceVec<long int> prevDelta;
	DeviceVec<long int> nextDelta;
	const char n = -2;
	const char z = 0;
	const char p = 2;

	prevDelta = DeviceVec<char>(n,n,n); nextDelta = DeviceVec<char>(p,p,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,z,n); nextDelta = DeviceVec<char>(p,z,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,p,n); nextDelta = DeviceVec<char>(p,n,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,p,n); nextDelta = DeviceVec<char>(z,n,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = DeviceVec<char>(n,n,z); nextDelta = DeviceVec<char>(p,p,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,z,z); nextDelta = DeviceVec<char>(p,z,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,p,z); nextDelta = DeviceVec<char>(p,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,p,z); nextDelta = DeviceVec<char>(p,p,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = DeviceVec<char>(n,n,p); nextDelta = DeviceVec<char>(p,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,z,p); nextDelta = DeviceVec<char>(p,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,p,p); nextDelta = DeviceVec<char>(p,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,p,p); nextDelta = DeviceVec<char>(z,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;


	prevDelta = DeviceVec<char>(n,n,n); nextDelta = DeviceVec<char>(n,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,z,n); nextDelta = DeviceVec<char>(n,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,p,n); nextDelta = DeviceVec<char>(n,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(n,p,z); nextDelta = DeviceVec<char>(n,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = DeviceVec<char>(z,n,n); nextDelta = DeviceVec<char>(z,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,z,n); nextDelta = DeviceVec<char>(z,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,p,n); nextDelta = DeviceVec<char>(z,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,p,z); nextDelta = DeviceVec<char>(z,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = DeviceVec<char>(p,n,n); nextDelta = DeviceVec<char>(p,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,z,n); nextDelta = DeviceVec<char>(p,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,p,n); nextDelta = DeviceVec<char>(p,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,p,z); nextDelta = DeviceVec<char>(p,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;


	prevDelta = DeviceVec<char>(n,n,n); nextDelta = DeviceVec<char>(p,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,n,n); nextDelta = DeviceVec<char>(z,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,n,n); nextDelta = DeviceVec<char>(n,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,n,z); nextDelta = DeviceVec<char>(n,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = DeviceVec<char>(n,z,n); nextDelta = DeviceVec<char>(p,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,z,n); nextDelta = DeviceVec<char>(z,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,z,n); nextDelta = DeviceVec<char>(n,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,z,z); nextDelta = DeviceVec<char>(n,z,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = DeviceVec<char>(n,p,n); nextDelta = DeviceVec<char>(p,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(z,p,n); nextDelta = DeviceVec<char>(z,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,p,n); nextDelta = DeviceVec<char>(n,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = DeviceVec<char>(p,p,z); nextDelta = DeviceVec<char>(n,p,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	return false;
}

template<class PixelType>
__device__ void evalNeighborhood(const CudaImageContainer<PixelType> &imageIn,const DeviceVec<size_t> &coordinate,double threshold,DeviceVec<size_t> hostKernelDims,CudaImageContainer<bool>& maskIn,CudaImageContainer<bool> &maskOut)
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

template<class PixelType>
__global__ void cudaRegionGrowing(CudaImageContainer<PixelType> imageIn,CudaImageContainer<bool> maskIn,CudaImageContainer<bool> maskOut,
	Vec<size_t> hostKernelDims,double threshold,bool allowConnection=true)
{
	DeviceVec<size_t> coordinate;
	DeviceVec<size_t> deviceKernelDims(hostKernelDims);
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
			if(!allowConnection)
			{
				if(willConnect(maskIn,coordinate))
					maskOut[coordinate] = false;
				else
					evalNeighborhood(imageIn,coordinate,threshold,deviceKernelDims,maskIn,maskOut);
			}
			else
			{
				evalNeighborhood(imageIn,coordinate,threshold,deviceKernelDims,maskIn,maskOut);
			}
		}
	}
}

template <class PixelType>
void cRegionGrowing(const PixelType* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0)
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
			*(deviceMask.getNextBuffer()),kernelDims,threshold,allowConnection);
		DEBUG_KERNEL_CHECK();

		curChunk->retriveROI(imageMask,dims,deviceMask.getNextBuffer());
	}
}
