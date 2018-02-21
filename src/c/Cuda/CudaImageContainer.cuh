#pragma once

#include "Vec.h"

#include <string>
#include "CudaUtilities.cuh"

template <class PixelType>
class CudaImageContainer
{
public:
	__host__ __device__ ~CudaImageContainer()
	{
		defaults();
	}

	__host__ __device__ CudaImageContainer(const CudaImageContainer& other)
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

	__device__ PixelType& operator()( Vec<size_t> coordinate )
	{
        return accessValue(coordinate);
	}

	__device__ const PixelType& operator()( Vec<size_t> coordinate ) const
	{
        return accessValue(coordinate);
	}

	__device__ const PixelType operator()(Vec<float> pos) const
	{
		Vec<size_t> curPos(0);
		double val = 0;
		if(pos.floor()==pos)
		{
			curPos = Vec<size_t>(pos);
			val = accessValue(curPos);
			return val;
		}

		// otherwise linear interpolation is needed
		// Math example from http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering

		Vec<float> betaPos = pos-0.5f;
		size_t i = floor(betaPos.x);
		size_t j = floor(betaPos.y);
		size_t k = floor(betaPos.z);
		float alpha = betaPos.x-floor(betaPos.x);
		float beta = betaPos.y-floor(betaPos.y);
		float gamma = betaPos.z-floor(betaPos.z);

		Vec<size_t> minPos(0, 0, 0);
		curPos = Vec<size_t>(i, j, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) * (1-beta)  * (1-gamma) * accessValue(curPos);

		curPos = Vec<size_t>(i+1, j, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += alpha  * (1-beta)  * (1-gamma) * accessValue(curPos);

		curPos = Vec<size_t>(i, j+1, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) *    beta   * (1-gamma) * accessValue(curPos);

		curPos = Vec<size_t>(i+1, j+1, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += alpha  *    beta   * (1-gamma) * accessValue(curPos);

		curPos = Vec<size_t>(i, j, k+1);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) * (1-beta)  *    gamma  * accessValue(curPos);

		curPos = Vec<size_t>(i+1, j, k+1);
		if(curPos>=minPos && curPos<roiSizes)
            val += alpha  * (1-beta)  *    gamma  * accessValue(curPos);

		curPos = Vec<size_t>(i, j+1, k+1);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) *    beta   *    gamma  * accessValue(curPos);

		curPos = Vec<size_t>(i+1, j+1, k+1);
		if(curPos>=minPos && curPos<roiSizes)
			val += alpha  *    beta   *    gamma  * accessValue(curPos);

		return val;
	}

	__device__ PixelType& operator[](size_t idx)
	{
        return accessValue(idx);
	}

	__device__ const PixelType& operator[]( size_t idx) const
	{
        return accessValue(idx);
	}

	__host__ __device__ const Vec<size_t>& getDims() const { return roiSizes; }
	__host__ __device__ size_t getWidth() const { return roiSizes.x; }
	__host__ __device__ size_t getHeight() const { return roiSizes.y; }
	__host__ __device__ size_t getDepth() const { return roiSizes.z; }

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

	__host__ __device__ void defaults()
	{
		maxImageDims = Vec<size_t>(0, 0, 0);
		imageDims = Vec<size_t>(0,0,0);
		device = 0;
		roiStarts = Vec<size_t>(0, 0, 0);
		roiSizes = Vec<size_t>(0, 0, 0);
	}

    __device__ PixelType& accessValue(Vec<size_t> coordinate)
    {
        return image[getIdx(coordinate)];
    }
    
    __device__ const PixelType& accessValue(Vec<size_t> coordinate) const
    {
        return image[getIdx(coordinate)];
    }

    __device__ PixelType& accessValue(size_t idx)
    {
        return image[getIdx(idx)];
    }
    
    __device__ const PixelType& accessValue(size_t idx) const
    {
        return image[getIdx(idx)];
    }

    __device__ size_t getIdx(Vec<size_t> coordinate) const
    {
        coordinate += Vec<size_t>(roiStarts);

        return imageDims.linearAddressAt(coordinate);
    }

    __device__ size_t getIdx(size_t idx) const
    {
        Vec<size_t> deviceStarts = Vec<size_t>(roiStarts);
        if(deviceStarts==Vec<size_t>(0, 0, 0))
            return idx;

        Vec<size_t> coordinate = imageDims.coordAddressOf(idx) + deviceStarts;

        return imageDims.linearAddressAt(coordinate);
    }

	int device;
	Vec<size_t> maxImageDims;
	Vec<size_t> imageDims;
	Vec<size_t> roiStarts;
	Vec<size_t> roiSizes;
	PixelType*	image;
};
