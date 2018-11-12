#pragma once

#include "CudaImageContainer.cuh"

template <class PixelType>
class CudaImageContainerClean : public CudaImageContainer<PixelType>
{
public:
	CudaImageContainerClean(const PixelType* imageIn, Vec<std::size_t> dims, int device = 0)
	{
		this->defaults();
		this->image = NULL;
		this->maxImageDims = dims;
		this->roiSizes = dims;
		this->device = device;
		this->loadImage(imageIn, dims);
	};

	CudaImageContainerClean(Vec<std::size_t> dims, int device = 0)
	{
		this->defaults();
		this->image = NULL;
		this->maxImageDims = dims;
		this->imageDims = dims;
		this->roiSizes = dims;
		this->device = device;
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaMalloc((void**)&this->image, sizeof(PixelType)*dims.product()));
		HANDLE_ERROR(cudaMemset(this->image, 0, sizeof(PixelType)*dims.product()));
	};

	~CudaImageContainerClean()
	{
		if (this->image != NULL)
		{
			HANDLE_ERROR(cudaSetDevice(this->device));
			try
			{
				HANDLE_ERROR(cudaFree(this->image));
			}
			catch (char* err)
			{
				if (err != NULL)
					err[0] = 'e';
			}
			this->image = NULL;
		}
	}

	CudaImageContainerClean(const CudaImageContainerClean& other)
	{
		this->device = other.getDeviceNumber();
		this->imageDims = other.getDims();
		this->image = NULL;

		HANDLE_ERROR(cudaSetDevice(this->device));

		if (this->imageDims > Vec<std::size_t>(0, 0, 0))
		{
			HANDLE_ERROR(cudaMalloc((void**)&this->image, sizeof(PixelType)*this->imageDims.product()));
			HANDLE_ERROR(cudaMemcpy(this->image, other.getConstImagePointer(), sizeof(PixelType)*this->imageDims.product(), cudaMemcpyDeviceToDevice));
		}
	}

protected:
	CudaImageContainerClean() : CudaImageContainer() {};
};