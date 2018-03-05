#pragma once

#include "CudaImageContainer.cuh"
#include "ImageDimensions.cuh"

template <class PixelType>
class CudaImageContainerClean : public CudaImageContainer<PixelType>
{
public:
	CudaImageContainerClean(const PixelType* imageIn, ImageDimensions dims, int device=0)
	{
		defaults();
		image = NULL;
		maxImageDims = dims;
		roiSizes = dims;
		this->device = device;
		loadImage(imageIn,dims);
	};

	CudaImageContainerClean(ImageDimensions dims, int device=0) 
	{
		defaults();
		image = NULL;
		maxImageDims = dims;
		imageDims = dims;
		roiSizes = dims;
		this->device = device;
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(PixelType)*dims.getNumElements()));
		HANDLE_ERROR(cudaMemset(image,0,sizeof(PixelType)*dims.getNumElements()));
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