#pragma once

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

__device__ bool lineConnect(const CudaImageContainer<bool>& maskIn,Vec<size_t> prevCoord,Vec<size_t> nextCoord)
{
	if(prevCoord>=Vec<size_t>(0,0,0) && nextCoord>=Vec<size_t>(0,0,0))
	{
		if(prevCoord<maskIn.getDims() && nextCoord<maskIn.getDims())
		{

			if(maskIn(Vec<size_t>(prevCoord)) && maskIn(Vec<size_t>(nextCoord)))
				return true;
		}
	}
	
	return false;
}

__device__ bool willConnect(const CudaImageContainer<bool>& maskIn, Vec<size_t> coordinateIn)
{
	if(maskIn(coordinateIn))
		return true;

	Vec<size_t> coordinate(coordinateIn);
	Vec<size_t> prevCoord;
	Vec<size_t> nextCoord;
	Vec<size_t> prevDelta;
	Vec<size_t> nextDelta;
	const char n = -2;
	const char z = 0;
	const char p = 2;

	prevDelta = Vec<char>(n,n,n); nextDelta = Vec<char>(p,p,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,z,n); nextDelta = Vec<char>(p,z,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,p,n); nextDelta = Vec<char>(p,n,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,p,n); nextDelta = Vec<char>(z,n,n);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = Vec<char>(n,n,z); nextDelta = Vec<char>(p,p,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,z,z); nextDelta = Vec<char>(p,z,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,p,z); nextDelta = Vec<char>(p,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,p,z); nextDelta = Vec<char>(p,p,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = Vec<char>(n,n,p); nextDelta = Vec<char>(p,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,z,p); nextDelta = Vec<char>(p,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,p,p); nextDelta = Vec<char>(p,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,p,p); nextDelta = Vec<char>(z,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;


	prevDelta = Vec<char>(n,n,n); nextDelta = Vec<char>(n,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,z,n); nextDelta = Vec<char>(n,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,p,n); nextDelta = Vec<char>(n,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(n,p,z); nextDelta = Vec<char>(n,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = Vec<char>(z,n,n); nextDelta = Vec<char>(z,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,z,n); nextDelta = Vec<char>(z,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,p,n); nextDelta = Vec<char>(z,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,p,z); nextDelta = Vec<char>(z,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = Vec<char>(p,n,n); nextDelta = Vec<char>(p,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,z,n); nextDelta = Vec<char>(p,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,p,n); nextDelta = Vec<char>(p,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,p,z); nextDelta = Vec<char>(p,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;


	prevDelta = Vec<char>(n,n,n); nextDelta = Vec<char>(p,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,n,n); nextDelta = Vec<char>(z,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,n,n); nextDelta = Vec<char>(n,n,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,n,z); nextDelta = Vec<char>(n,n,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = Vec<char>(n,z,n); nextDelta = Vec<char>(p,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,z,n); nextDelta = Vec<char>(z,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,z,n); nextDelta = Vec<char>(n,z,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,z,z); nextDelta = Vec<char>(n,z,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	prevDelta = Vec<char>(n,p,n); nextDelta = Vec<char>(p,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(z,p,n); nextDelta = Vec<char>(z,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,p,n); nextDelta = Vec<char>(n,p,p);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;
	prevDelta = Vec<char>(p,p,z); nextDelta = Vec<char>(n,p,z);
	if(lineConnect(maskIn,coordinate+prevDelta,coordinate+nextDelta)) return true;

	return false;
}

template<class PixelType>
__device__ void evalNeighborhood(const CudaImageContainer<PixelType> &imageIn,const Vec<size_t> &coordinate,double threshold,Vec<size_t> hostKernelDims,CudaImageContainer<bool>& maskIn,CudaImageContainer<bool> &maskOut)
{
	PixelType curPixelVal = imageIn(coordinate) + threshold;
	Vec<size_t> kernelDims = hostKernelDims;
	Vec<size_t> halfKernal = kernelDims/2;

	Vec<size_t> curCoordIm = coordinate - halfKernal;
	curCoordIm.z = (coordinate.z<halfKernal.z) ? 0 : coordinate.z-halfKernal.z;
	for(; curCoordIm.z<=coordinate.z+halfKernal.z && curCoordIm.z<imageIn.getDims().z; ++curCoordIm.z)
	{
		curCoordIm.y = (coordinate.y<halfKernal.y) ? 0 : coordinate.y-halfKernal.y;
		for(; curCoordIm.y<=coordinate.y+halfKernal.y && curCoordIm.y<imageIn.getDims().y; ++curCoordIm.y)
		{
			curCoordIm.x = (coordinate.x<halfKernal.x) ? 0 : coordinate.x-halfKernal.x;
			for(; curCoordIm.x<=coordinate.x+halfKernal.x && curCoordIm.x<imageIn.getDims().x; ++curCoordIm.x)
			{
				if(curPixelVal > imageIn(curCoordIm) && maskIn(curCoordIm)==true)
				{
					maskOut(coordinate) = true;
				}
			}
		}
	}
}

template<class PixelType>
__global__ void cudaRegionGrowing(CudaImageContainer<PixelType> imageIn,CudaImageContainer<bool> maskIn,CudaImageContainer<bool> maskOut,
	Vec<size_t> hostKernelDims,double threshold,bool allowConnection=true)
{
	Vec<size_t> coordinate;
	Vec<size_t> deviceKernelDims(hostKernelDims);
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if(coordinate<imageIn.getDims())
	{
		if(maskIn(coordinate)==true)
		{
			maskOut(coordinate) = true;
		}
		else
		{
			if(!allowConnection)
			{
				if(willConnect(maskIn,coordinate))
					maskOut(coordinate) = false;
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
	//cudaSetDevice(device);
	//PixelType minVal = std::numeric_limits<PixelType>::lowest();
	//PixelType maxVal = std::numeric_limits<PixelType>::max();

	//if(kernel==NULL)
	//{
	//	kernelDims = kernelDims.clamp(Vec<size_t>(1,1,1),dims);
	//	float* ones = new float[kernelDims.product()];
	//	for(int i=0; i<kernelDims.product(); ++i)
	//		ones[i] = 1.0f;

	//	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,ones,sizeof(float)*kernelDims.product()));
	//	delete[] ones;
	//} else
	//{
	//	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,kernel,sizeof(float)*kernelDims.product()));
	//}

	//cudaDeviceProp props;
	//cudaGetDeviceProperties(&props,device);

	//size_t availMem,total;
	//cudaMemGetInfo(&availMem,&total);

	//std::vector<ImageChunk> chunks = calculateBuffers<PixelType>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,kernelDims);

	//Vec<size_t> maxDeviceDims;
	//setMaxDeviceDims(chunks,maxDeviceDims);

	//CudaDeviceImages<PixelType> deviceImages(1,maxDeviceDims,device);
	//CudaDeviceImages<bool> deviceMask(2,maxDeviceDims,device);

	//for(std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	//{
	//	curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
	//	deviceImages.setNextDims(curChunk->getFullChunkSize());
	//	curChunk->sendROI(imageMask,dims,deviceMask.getCurBuffer());
	//	deviceMask.setNextDims(curChunk->getFullChunkSize());

	//	cudaRegionGrowing<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceMask.getCurBuffer()),
	//		*(deviceMask.getNextBuffer()),kernelDims,threshold,allowConnection);
	//	DEBUG_KERNEL_CHECK();

	//	curChunk->retriveROI(imageMask,dims,deviceMask.getNextBuffer());
	//}
}
