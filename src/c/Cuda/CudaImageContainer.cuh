#pragma once

#include "Vec.h"
#include "CudaUtilities.h"
#include "ImageContainer.h"

#include <string>


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

	PixelType* getDeviceImagePointer()const {return image;}

	__device__ PixelType* getImagePointer(){return image;}

	__device__ PixelType& operator()(const ImageDimensions coordinate)
	{
        return accessValue(coordinate);
	}

	__device__ const PixelType& operator()( ImageDimensions coordinate) const
	{
        return accessValue(coordinate);
	}

	__device__ const PixelType operator()(Vec<float> pos, unsigned int chan=0, unsigned	int frame=0) const
	{
		Vec<size_t> curPos(0);
		double val = 0;
		if(pos.floor()==pos)
		{
			curPos = Vec<size_t>(pos);
			val = accessValue(curPos,chan,frame);
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
			val += (1-alpha) * (1-beta)  * (1-gamma) * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i+1, j, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += alpha  * (1-beta)  * (1-gamma) * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i, j+1, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) *    beta   * (1-gamma) * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i+1, j+1, k);
		if(curPos>=minPos && curPos<roiSizes)
			val += alpha  *    beta   * (1-gamma) * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i, j, k+1);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) * (1-beta)  *    gamma  * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i+1, j, k+1);
		if(curPos>=minPos && curPos<roiSizes)
            val += alpha  * (1-beta)  *    gamma  * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i, j+1, k+1);
		if(curPos>=minPos && curPos<roiSizes)
			val += (1-alpha) *    beta   *    gamma  * accessValue(curPos, chan, frame);

		curPos = Vec<size_t>(i+1, j+1, k+1);
		if(curPos>=minPos && curPos<roiSizes)
			val += alpha  *    beta   *    gamma  * accessValue(curPos, chan, frame);

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

	__host__ __device__ const ImageDimensions& getDims() const { return imageDims; }
	__host__ __device__ const Vec<size_t> getSpatialDims() const { return roiSizes; }
	__host__ __device__ size_t getWidth() const { return roiSizes.x; }
	__host__ __device__ size_t getHeight() const { return roiSizes.y; }
	__host__ __device__ size_t getDepth() const { return roiSizes.z; }
	__host__ __device__ unsigned int getNumChans() const { return imageDims.chan; }
	__host__ __device__ unsigned int getNumFrames() const { return imageDims.frame; }

	bool setDims(ImageDimensions dims)
	{
		if (maxImageDims.getNumElements()<dims.getNumElements())
			return false;

		imageDims = dims;

		return true;
	}
	bool setROIstart(Vec<size_t> starts)
	{
		if(starts>=imageDims.dims)
			return false;

		roiStarts = starts;
		return true;
	}
	bool setROIsize(Vec<size_t> sizes)
	{
		if(roiStarts+sizes>imageDims.dims)
			return false;

		roiSizes = sizes;
		return true;
	}
	void resetROI()
	{
		roiStarts = Vec<size_t>(0,0,0);
		roiSizes = imageDims.dims;
	}

	void loadImage(const ImageContainer<PixelType> imageIn)
	{
		HANDLE_ERROR(cudaSetDevice(device));
		if(imageIn.getDims()!=imageDims)
		{
			if(image!=NULL)
			{
				HANDLE_ERROR(cudaFree(image));
			}
			checkFreeMemory(sizeof(PixelType)*imageIn.getDims().getNumElements(),device,true);
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*imageIn.getDims().getNumElements()));
			imageDims = imageIn.getDims();
		}

		//const void* imPtr = (void*)imageIn.getConstPtr();
		size_t numEl = sizeof(PixelType)*imageIn.getDims().getNumElements();
		const void* imIn = imageIn.getConstPtr();
		HANDLE_ERROR(cudaMemcpy(image, imIn, numEl, cudaMemcpyHostToDevice));
	}

protected:
	CudaImageContainer()
	{
		defaults();
		image = NULL;
	}

	CudaImageContainer(const PixelType* imageIn,ImageDimensions dims,int device=0) {}

	CudaImageContainer(ImageDimensions dims,int device=0) {}

	__host__ __device__ void defaults()
	{
		maxImageDims = ImageDimensions(Vec<size_t>(0), 0, 0);
		imageDims = ImageDimensions(Vec<size_t>(0), 0, 0);
		device = 0;
		roiStarts = Vec<size_t>(0);
		roiSizes = Vec<size_t>(0);
	}

    __device__ PixelType& accessValue(Vec<size_t> coordinate, unsigned int chan = 0, unsigned int frame = 0)
    {
		ImageDimensions curCoord(coordinate, chan, frame);
        return image[getIdx(curCoord)];
    }

	__device__ PixelType& accessValue(ImageDimensions coordinate) const
	{
		return image[getIdx(coordinate)];
	}

	__device__ const PixelType& accessValue(Vec<size_t> dims, unsigned int chan, unsigned int frame) const
	{
		ImageDimensions coordinate(dims, chan, frame);
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

    __device__ size_t getIdx(ImageDimensions coordinate) const
    {
        coordinate += roiStarts;

        return imageDims.linearAddressAt(coordinate);
    }

    __device__ size_t getIdx(size_t idx) const
    {
        if(roiStarts==Vec<size_t>(0))
            return idx;

		ImageDimensions coordinate(imageDims.coordAddressOf(idx));
		coordinate.dims	+= roiStarts;

        return imageDims.linearAddressAt(coordinate);
    }

	int device;
	ImageDimensions maxImageDims;
	ImageDimensions imageDims;
	Vec<size_t> roiStarts;
	Vec<size_t> roiSizes;
	PixelType*	image;
};

template <class PixelType>
void setUpOutIm(ImageDimensions dims, ImageContainer<PixelType> imageOut)
{
	if (imageOut.getPtr() == NULL)
	{
		imageOut.resize(dims);
	}
}
