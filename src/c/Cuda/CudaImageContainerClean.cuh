#pragma once

#include "CudaImageContainer.cuh"
#include "ImageDimensions.cuh"

template <class PixelType>
class CudaImageContainerClean : public CudaImageContainer<PixelType>
{
public:
	CudaImageContainerClean(const PixelType* imageIn, ImageDimensions imDims, int device=0)
	{
		defaults();
		image = NULL;
		maxImageDims = imDims;
		roiSizes = imDims.dims;
		this->device = device;
		ImageContainer<PixelType> im(imageIn, imDims);
		loadImage(im);
	};

	CudaImageContainerClean(ImageDimensions imDims, int device=0) 
	{
		defaults();
		image = NULL;
		maxImageDims = imDims;
		imageDims = imDims;
		roiSizes = imDims.dims;
		this->device = device;
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*imDims.getNumElements()));
		HANDLE_ERROR(cudaMemset(image,0,sizeof(PixelType)*imDims.getNumElements()));
	};

	~CudaImageContainerClean()
	{
		if (image!=NULL)
		{
			HANDLE_ERROR(cudaSetDevice(device));
			try
			{
				HANDLE_ERROR(cudaFree(image));
			}
			catch (char* err)
			{
				if (err!=NULL)
					err[0] = 'e';
			}
			image = NULL;
		}
	}

	CudaImageContainerClean(const CudaImageContainerClean& other)
	{
		device = other.getDeviceNumber();
		imageDims = other.getDims();
		image = NULL;

		HANDLE_ERROR(cudaSetDevice(device));

		if (imageDims.getNumElements()>0)
		{
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*imageDims.getNumElements()));
			HANDLE_ERROR(cudaMemcpy(image,other.getConstImagePointer(),sizeof(PixelType)*imageDims.getNumElements(),cudaMemcpyDeviceToDevice));
		}
	}

protected:
	CudaImageContainerClean() : CudaImageContainer(){};
};