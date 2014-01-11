#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include <string>
#include "CudaUtilities.cuh"

typedef unsigned char DevicePixelType;

class CudaImageContainer
{
public:
	CudaImageContainer( DevicePixelType* imageIn, Vec<size_t> dims, int device=0)
	{
		defaults();
		image = NULL;
		loadImage(imageIn, dims);
	}

	CudaImageContainer(Vec<size_t> dims, int device=0)
	{
		defaults();
		imageDims = dims;
		HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(DevicePixelType)*dims.product()));
	}

	~CudaImageContainer()
	{
		defaults();
	}

	CudaImageContainer(const CudaImageContainer& other)
	{
		device = other.device;
		imageDims = other.imageDims;
		image = other.image;
	}

	const DevicePixelType* getConstImagePointer() const {return image;}

	DevicePixelType* getImagePointer(){return image;}

	int getDeviceNumber() const {return device;}

	__device__ DevicePixelType* getDeviceImagePointer(){return image;}

	__device__ DevicePixelType& operator[]( DeviceVec<size_t> coordinate )
	{
		const DeviceVec<size_t>& deviceImageDims = reinterpret_cast<const DeviceVec<size_t>&>(imageDims);
		return image[deviceImageDims.linearAddressAt(coordinate)];
	}

	__device__ const DevicePixelType& operator[]( DeviceVec<size_t> coordinate ) const
	{
		const DeviceVec<size_t>& deviceImageDims = reinterpret_cast<const DeviceVec<size_t>&>(imageDims);
		return image[deviceImageDims.linearAddressAt(coordinate)];
	}

	__device__ DevicePixelType& operator[](size_t idx)
	{
		return image[idx];
	}

	__device__ const DevicePixelType& operator[]( size_t idx) const
	{
		return image[idx];
	}

	void loadImage( const DevicePixelType* imageIn, Vec<size_t> dims)
	{
		HANDLE_ERROR(cudaSetDevice(device));
		if (dims!=imageDims)
		{
			if (image!=NULL)
			{
				HANDLE_ERROR(cudaFree(image));
			}
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(DevicePixelType)*dims.product()));
			imageDims = dims;
		}

		HANDLE_ERROR(cudaMemcpy(image,imageIn,sizeof(DevicePixelType)*dims.product(),cudaMemcpyHostToDevice));
	}

	Vec<size_t> getDims() const {return Vec<size_t>(imageDims);}
	__device__ DeviceVec<size_t> getDeviceDims() const {return DeviceVec<size_t>(imageDims);}
	__device__ size_t getWidth() const {return imageDims.x;}
	__device__ size_t getHeight() const {return imageDims.y;}
	__device__ size_t getDepth() const {return imageDims.z;}

protected:
	CudaImageContainer();

	void defaults() 
	{
		imageDims = Vec<size_t>(0,0,0);
		device = 0;
	}

	int device;
	Vec<size_t> imageDims;
	DevicePixelType*	image;
};