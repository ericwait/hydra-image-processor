#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include <string>
#include "CudaUtilities.cuh"

typedef unsigned char DevicePixelType;

class CudaImageContainer
{
public:
// 	CudaImageContainer(size_t width, size_t height, size_t depth, bool isColumnMajor=false)
// 	{
// 		defaults();
// 		imageDims.x = width;
// 		imageDims.y = height;
// 		imageDims.z = depth;
// 
// 		columnMajor = isColumnMajor;
// 
// 		image = new DevicePixelType[imageDims.product()];
// 	}
// 
// 	CudaImageContainer(Vec<size_t> dimsIn, bool isColumnMajor=false)
// 	{
// 		defaults();
// 		imageDims = dimsIn;
// 
// 		columnMajor = isColumnMajor;
// 
// 		image = new DevicePixelType[imageDims.product()];
// 	}

	CudaImageContainer( DevicePixelType* imageIn, Vec<size_t> dims, int device=0, bool isColumnMajor=false)
	{
		defaults();
		columnMajor = isColumnMajor;
		loadImage(imageIn, dims);
	}

	CudaImageContainer(Vec<size_t> dims, int device=0, bool isColumnMajor=false)
	{
		defaults();
		columnMajor = isColumnMajor;
		if (dims!=imageDims)
		{
			if (image!=NULL)
			{
				HANDLE_ERROR(cudaFree(image));
			}
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(DevicePixelType)*dims.product()));
			imageDims = dims;
		}
	}

	~CudaImageContainer()
	{
		clear();
	}

	CudaImageContainer(const CudaImageContainer& other)
	{
		device = other.device;
		imageDims = other.imageDims;
		columnMajor = other.columnMajor;

		HANDLE_ERROR(cudaSetDevice(device));

		HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(DevicePixelType)*imageDims.product()));
		HANDLE_ERROR(cudaMemcpy(image,other.image,sizeof(DevicePixelType)*imageDims.product(),cudaMemcpyDeviceToDevice));
	}

	const DevicePixelType* getConstImagePointer() const {return image;}

	DevicePixelType* getImagePointer(){return image;}

	__device__ DevicePixelType& operator[]( DeviceVec<size_t> coordinate )
	{
		const DeviceVec<size_t>& deviceImageDims = reinterpret_cast<const DeviceVec<size_t>&>(imageDims);
		return image[deviceImageDims.linearAddressAt(coordinate,columnMajor)];
	}

	__device__ const DevicePixelType& operator[]( DeviceVec<size_t> coordinate ) const
	{
		const DeviceVec<size_t>& deviceImageDims = reinterpret_cast<const DeviceVec<size_t>&>(imageDims);
		return image[deviceImageDims.linearAddressAt(coordinate,columnMajor)];
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

	__device__ DeviceVec<size_t> getDims() const {return DeviceVec<size_t>(imageDims);}
	__device__ size_t getWidth() const {return imageDims.x;}
	__device__ size_t getHeight() const {return imageDims.y;}
	__device__ size_t getDepth() const {return imageDims.z;}

private:
	CudaImageContainer();

	void clear()
	{
		if (image!=NULL)
		{
			HANDLE_ERROR(cudaFree(image));
			image = NULL;
		}

		defaults();
	}

	void defaults() 
	{
		imageDims = Vec<size_t>(0,0,0);
		columnMajor = false;
		image = NULL;
		device = 0;
	}

	int device;
	Vec<size_t> imageDims;
	bool columnMajor;
	DevicePixelType*	image;
};