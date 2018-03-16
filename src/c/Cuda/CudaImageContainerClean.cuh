#pragma once

#include "CudaImageContainer.cuh"

template <class PixelType>
class CudaImageContainerClean : public CudaImageContainer<PixelType>
{
public:
	CudaImageContainerClean(const PixelType* imageIn, Vec<size_t> dims, int device = 0)
	{
		defaults();
		image = NULL;
		maxImageDims = dims;
		roiSizes = dims;
		this->device = device;
		loadImage(imageIn, dims);
	};

	CudaImageContainerClean(Vec<size_t> dims, int device = 0)
	{
		defaults();
		image = NULL;
		maxImageDims = dims;
		imageDims = dims;
		roiSizes = dims;
		this->device = device;
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaMalloc((void**)&image, sizeof(PixelType)*dims.product()));
		HANDLE_ERROR(cudaMemset(image, 0, sizeof(PixelType)*dims.product()));
	};

	~CudaImageContainerClean()
	{
		if (image != NULL)
		{
			HANDLE_ERROR(cudaSetDevice(device));
			try
			{
				HANDLE_ERROR(cudaFree(image));
			}
			catch (char* err)
			{
				if (err != NULL)
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

		if (imageDims > Vec<size_t>(0, 0, 0))
		{
			HANDLE_ERROR(cudaMalloc((void**)&image, sizeof(PixelType)*imageDims.product()));
			HANDLE_ERROR(cudaMemcpy(image, other.getConstImagePointer(), sizeof(PixelType)*imageDims.product(), cudaMemcpyDeviceToDevice));
		}
	}

protected:
	CudaImageContainerClean() : CudaImageContainer() {};
};