#pragma once

#include "CudaImageContainer.cuh"

class CudaImageContainerClean : public CudaImageContainer
{
public:
	CudaImageContainerClean(const DevicePixelType* imageIn, Vec<size_t> dims, int device=0) : CudaImageContainer(imageIn,dims,device){};
	CudaImageContainerClean(Vec<size_t> dims, int device=0) : CudaImageContainer(dims,device){};
	CudaImageContainerClean(const ImageContainer* imageIn) : CudaImageContainer(imageIn){};

	~CudaImageContainerClean()
	{
		if (image!=NULL)
		{
			try
			{
				HANDLE_ERROR(cudaFree(image));
			}
			catch (char* err)
			{
				;
			}
			image = NULL;
		}
	}

	CudaImageContainerClean(const CudaImageContainer& other)
	{
		device = other.getDeviceNumber();
		imageDims = other.getDims();
		image = NULL;

		HANDLE_ERROR(cudaSetDevice(device));

		if (imageDims>Vec<size_t>(0,0,0))
		{
			HANDLE_ERROR(cudaMalloc((void**)&image,sizeof(DevicePixelType)*imageDims.product()));
			HANDLE_ERROR(cudaMemcpy(image,other.getConstImagePointer(),sizeof(DevicePixelType)*imageDims.product(),cudaMemcpyDeviceToDevice));
		}
	};
private:
	CudaImageContainerClean() : CudaImageContainer(){};
};