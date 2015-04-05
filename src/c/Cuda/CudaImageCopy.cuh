#pragma once
#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "CudaImageContainer.cuh"

#include "Vec.h"
#include "ImageChunk.cuh"

template <class PixelType>
__global__ void cudaImageCopy(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if(coordinate<imageIn.getDeviceDims() && coordinate<imageOut.getDeviceDims())
	{
		imageOut[coordinate] = imageIn[coordinate];
	}
}

template <class PixelType>
PixelType* cTileImage(const PixelType* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize, PixelType** imageOut=NULL,int device=0)
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	CudaDeviceImages<PixelType> deviceImages(2,dims,device);

	dim3 blocks,threads;
	calcBlockThread(roiSize,props,blocks,threads);

	deviceImages.getCurBuffer()->loadImage(imageIn,dims);
	deviceImages.getCurBuffer()->setROIstart(roiStart);
	deviceImages.getCurBuffer()->setROIsize(roiSize);

	Vec<size_t> numTiles;
	numTiles.x = size_t(ceil(double(dims.x)/double(roiSize.x)));
	numTiles.y = size_t(ceil(double(dims.y)/double(roiSize.y)));
	numTiles.z = size_t(ceil(double(dims.z)/double(roiSize.z)));

	Vec<size_t> curTile(0,0,0);
	Vec<size_t> curStart(0,0,0);
	Vec<size_t> curSize(roiSize);
	for(curTile.z=0; curTile.z<numTiles.z; ++curTile.z)
	{
		curStart.z = curTile.z*roiSize.z;
		curSize.z = MIN(dims.z-curStart.z,roiSize.z);
		
		for (curTile.y=0; curTile.y<numTiles.y; ++curTile.y)
		{
			curStart.y = curTile.y*roiSize.y;
			curSize.y = MIN(dims.y-curStart.y,roiSize.y);
			
			for (curTile.x=0; curTile.x<numTiles.x; ++curTile.x)
			{
				curStart.x = curTile.x*roiSize.x;
				curSize.x = MIN(dims.x-curStart.x,roiSize.x);

				if(!deviceImages.getNextBuffer()->setROIstart(curStart))
				{
					char buffer[255];
					deviceImages.getNextBuffer()->resetROI();
					Vec<size_t> d = deviceImages.getNextBuffer()->getDims();
					sprintf_s(buffer,"Trying to set a start of (%d,%d,%d), when the size is (%d,%d,%d)",
						curStart.x,curStart.y,curStart.z,d.x,d.y,d.z);
					throw std::runtime_error(buffer);
				}
				
				if(!deviceImages.getNextBuffer()->setROIsize(curSize))
				{
					char buffer[255];
					deviceImages.getNextBuffer()->resetROI();
					Vec<size_t> d = deviceImages.getNextBuffer()->getDims()-curStart;
					sprintf_s(buffer,"Trying to set a size of (%d,%d,%d), when the size to the end is (%d,%d,%d)",
						curSize.x,curSize.y,curSize.z,d.x,d.y,d.z);
					throw std::runtime_error(buffer);
				}

				cudaImageCopy<<<blocks,threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()));
				DEBUG_KERNEL_CHECK();
			}
		}
	}

	deviceImages.getNextBuffer()->resetROI();

	return deviceImages.getNextBuffer()->retriveImage(imageOut);
}