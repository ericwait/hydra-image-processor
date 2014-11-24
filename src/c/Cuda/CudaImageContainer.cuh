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
	CudaImageContainer(const PixelType* imageIn, Vec<size_t> dims, int device=0)
	{
		defaults();
		image = NULL;
		maxImageDims = dims;
		roiSizes = dims;
		this->device = device;
		loadImage(imageIn, dims);
	}

	CudaImageContainer(Vec<size_t> dims, int device=0)
	{
		defaults();
		image = NULL;
		imageDims = dims;
		maxImageDims = dims;
		roiSizes = dims;
		this->device = device;
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*dims.product()));
	}

	~CudaImageContainer()
	{
		defaults();
	}

	CudaImageContainer(const CudaImageContainer& other)
	{
		device = other.device;
		imageDims = other.imageDims;
		maxImageDims = other.maxImageDims;
		image = other.image;
	}

	const PixelType* getConstImagePointer() const {return image;}

	int getDeviceNumber() const {return device;}

	PixelType* getDeviceImagePointer(){return image;}

	__device__ PixelType* getImagePointer(){return image;}

	__device__ PixelType& operator[]( DeviceVec<size_t> coordinate )
	{
		coordinate += roiStarts;
		const DeviceVec<size_t>& deviceImageDims = reinterpret_cast<const DeviceVec<size_t>&>(imageDims);
		return image[deviceImageDims.linearAddressAt(coordinate)];
	}

	__device__ const PixelType& operator[]( DeviceVec<size_t> coordinate ) const
	{
		coordinate += roiStarts;
		const DeviceVec<size_t>& deviceImageDims = reinterpret_cast<const DeviceVec<size_t>&>(imageDims);
		return image[deviceImageDims.linearAddressAt(coordinate)];
	}

	__device__ PixelType& operator[](size_t idx)
	{
		if (roiStarts==DeviceVec<size_t> (0,0,0))
			return image[idx];

		DeviceVec<size_t> coord = imageDims.coordAddressOf (idx);
		coord += roiStarts;

		if (coord>roiStarts+roiSizes)
			throw runtime_error ("Index is out of ROI bounds!");

		return this[coord];
	}

	__device__ const PixelType& operator[]( size_t idx) const
	{
		if (roiStarts==DeviceVec<size_t> (0,0,0))
			return image[idx];

		DeviceVec<size_t> coord = imageDims.coordAddressOf (idx);
		coord += roiStarts;

		if (coord>roiStarts+roiSizes)
			throw runtime_error ("Index is out of ROI bounds!");

		return this[coord];
	}

	void loadImage( const PixelType* imageIn, Vec<size_t> dims)
	{
		HANDLE_ERROR(cudaSetDevice(device));
		if (dims!=imageDims)
		{
			if (image!=NULL)
			{
				HANDLE_ERROR(cudaFree(image));
			}
			checkFreeMemory(sizeof(PixelType)*dims.product(),device,true);
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*dims.product()));
			imageDims = dims;
		}

		HANDLE_ERROR(cudaMemcpy(image,imageIn,sizeof(PixelType)*dims.product(),cudaMemcpyHostToDevice));
	}

	Vec<size_t> getDims() const {return Vec<size_t>(imageDims);}
	__device__ DeviceVec<size_t> getDeviceDims() const {return DeviceVec<size_t>(imageDims);}
	__device__ size_t getWidth() const {return imageDims.x;}
	__device__ size_t getHeight() const {return imageDims.y;}
	__device__ size_t getDepth() const {return imageDims.z;}

	bool setDims(Vec<size_t> dims)
	{
		if (maxImageDims.product()<dims.product())
			return false;

		imageDims = dims;

		return true;
	}

protected:
	CudaImageContainer()
	{
		defaults();
		image = NULL;
	}

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
	PixelType*	image;
	Vec<size_t> roiStarts;
	Vec<size_t> roiSizes;
};
