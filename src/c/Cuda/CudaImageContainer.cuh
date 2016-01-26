#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "Vec.h"

#include <string>
#include "CudaUtilities.cuh"

template <class PixelType>
class CudaImageContainer
{
public:
	~CudaImageContainer()
	{
		defaults();
	}

	CudaImageContainer(const CudaImageContainer& other)
	{
		device = other.device;
		maxImageDims = other.maxImageDims;
		imageDims = other.imageDims;
		roiStarts = other.roiStarts;
		roiSizes = other.roiSizes;
		image = other.image;
	}

	const PixelType* getConstImagePointer() const {return image;}

	int getDeviceNumber() const {return device;}

	PixelType* getDeviceImagePointer(){return image;}

	__device__ PixelType* getImagePointer(){return image;}

	__device__ PixelType& operator[]( DeviceVec<size_t> coordinate )
	{
		coordinate += DeviceVec<size_t>(roiStarts);
		DeviceVec<size_t> deviceImageDims = DeviceVec<size_t>(imageDims);

		size_t ind = deviceImageDims.linearAddressAt(coordinate);
		return image[ind];
	}

	__device__ const PixelType& operator[]( DeviceVec<size_t> coordinate ) const
	{
		coordinate += DeviceVec<size_t>(roiStarts);
		DeviceVec<size_t> deviceImageDims = DeviceVec<size_t>(imageDims);

		return image[deviceImageDims.linearAddressAt(coordinate)];
	}

	__device__ PixelType& operator[](size_t idx)
	{
		DeviceVec<size_t> deviceStarts = DeviceVec<size_t>(roiStarts);
		if(deviceStarts==DeviceVec<size_t>(0,0,0))
			return image[idx];

		DeviceVec<size_t> deviceImageDims = DeviceVec<size_t>(imageDims);
		DeviceVec<size_t> coord = deviceImageDims.coordAddressOf(idx);

		coord += deviceStarts;
		return this[coord];
	}

	__device__ const PixelType& operator[]( size_t idx) const
	{
		DeviceVec<size_t> deviceStarts = DeviceVec<size_t>(roiStarts);
		if(deviceStarts==DeviceVec<size_t>(0,0,0))
			return image[idx];

		DeviceVec<size_t> deviceImageDims = DeviceVec<size_t>(imageDims);
		DeviceVec<size_t> coord = deviceImageDims.coordAddressOf(idx);

		coord += deviceStarts;
		return this[coord];
	}

	Vec<size_t> getDims() const { return roiSizes; }
	__device__ DeviceVec<size_t> getDeviceDims() const { return DeviceVec<size_t>(roiSizes); }
	__device__ size_t getWidth() const { return DeviceVec<size_t>(roiSizes).x; }
	__device__ size_t getHeight() const { return DeviceVec<size_t>(roiSizes).y; }
	__device__ size_t getDepth() const { return DeviceVec<size_t>(roiSizes).z; }

	bool setDims(Vec<size_t> dims)
	{
		if (maxImageDims.product()<dims.product())
			return false;

		imageDims = dims;

		return true;
	}
	bool setROIstart(Vec<size_t> starts)
	{
		if(starts>=imageDims)
			return false;

		roiStarts = starts;
		return true;
	}
	bool setROIsize(Vec<size_t> sizes)
	{
		if(roiStarts+sizes>imageDims)
			return false;

		roiSizes = sizes;
		return true;
	}
	void resetROI()
	{
		roiStarts = Vec<size_t>(0,0,0);
		roiSizes = imageDims;
	}

	void loadImage(const PixelType* imageIn,Vec<size_t> dims)
	{
		HANDLE_ERROR(cudaSetDevice(device));
		if(dims!=imageDims)
		{
			if(image!=NULL)
			{
				HANDLE_ERROR(cudaFree(image));
			}
			checkFreeMemory(sizeof(PixelType)*dims.product(),device,true);
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*dims.product()));
			imageDims = dims;
		}

		HANDLE_ERROR(cudaMemcpy(image,imageIn,sizeof(PixelType)*dims.product(),cudaMemcpyHostToDevice));
	}

protected:
	CudaImageContainer()
	{
		defaults();
		image = NULL;
	}

	CudaImageContainer(const PixelType* imageIn,Vec<size_t> dims,int device=0) {}

	CudaImageContainer(Vec<size_t> dims,int device=0) {}

	void defaults() 
	{
		maxImageDims = Vec<size_t>(0, 0, 0);
		imageDims = Vec<size_t>(0,0,0);
		device = 0;
		roiStarts = Vec<size_t>(0, 0, 0);
		roiSizes = Vec<size_t>(0, 0, 0);
	}

	int device;
	Vec<size_t> maxImageDims;
	Vec<size_t> imageDims;
	Vec<size_t> roiStarts;
	Vec<size_t> roiSizes;
	PixelType*	image;
};
