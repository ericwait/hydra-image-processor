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

	PixelType* getDeviceImagePointer(){return image;}

	__device__ PixelType* getImagePointer(){return image;}

	__device__ PixelType& operator()( ImageDimensions coordinate)
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

	__host__ __device__ const ImageDimensions& getDims() const { return roiSizes; }
	__host__ __device__ const Vec<size_t> getSpatialDims() const { return roiSizes.dims; }
	__host__ __device__ size_t getWidth() const { return roiSizes.dims.x; }
	__host__ __device__ size_t getHeight() const { return roiSizes.dims.y; }
	__host__ __device__ size_t getDepth() const { return roiSizes.dims.z; }
	__host__ __device__ unsigned int getNumChans() const { return roiSizes.chan; }
	__host__ __device__ unsigned int getNumFrames() const { return roiSizes.frame; }

	bool setDims(ImageDimensions dims)
	{
		if (maxImageDims.getNumElements()<dims.getNumElements())
			return false;

		imageDims = dims;

		return true;
	}
	bool setROIstart(ImageDimensions starts)
	{
		if(starts>=imageDims)
			return false;

		roiStarts = starts;
		return true;
	}
	bool setROIsize(ImageDimensions sizes)
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

	void loadImage(const ImageContainer<PixelType> imageIn)
	{
		HANDLE_ERROR(cudaSetDevice(device));
		if(imageIn.dims!=imageDims)
		{
			if(image!=NULL)
			{
				HANDLE_ERROR(cudaFree(image));
			}
			checkFreeMemory(sizeof(PixelType)*imageIn.dims.getNumElements(),device,true);
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*imageIn.dims.getNumElements()));
			imageDims = imageIn.dims;
		}

		HANDLE_ERROR(cudaMemcpy(image,imageIn.getConstPtr(),sizeof(PixelType)*imageIn.dims.getNumElements(),cudaMemcpyHostToDevice));
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
		roiStarts = ImageDimensions(Vec<size_t>(0), 0, 0);
		roiSizes = ImageDimensions(Vec<size_t>(0), 0, 0);
	}

    __device__ PixelType& accessValue(Vec<size_t> coordinate, unsigned int chan = 0, unsigned int frame = 0)
    {
		ImageDimensions curCoord(coordinate, chan, frame);
        return image[getIdx(curCoord)];
    }
    
    __device__ const PixelType& accessValue(ImageDimensions coordinate) const
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
        if(roiStarts==ImageDimensions(0,0,0))
            return idx;

        ImageDimensions coordinate = imageDims.coordAddressOf(idx) + roiStarts;

        return imageDims.linearAddressAt(coordinate);
    }

	int device;
	ImageDimensions maxImageDims;
	ImageDimensions imageDims;
	ImageDimensions roiStarts;
	ImageDimensions roiSizes;
	PixelType*	image;
};

template <class PixelType>
ImageContainer<PixelType> setUpOutIm(Vec<size_t> dims, PixelType** imageOut)
{

	PixelType* imOut;
	if (imageOut == NULL)
		imOut = new PixelType[dims.product()];
	else
		imOut = *imageOut;

	return imOut;
}