#pragma once

#include "CudaUtilities.cuh"
#include "CudaKernels.cuh"
#include "Vec.h"

template<typename ImagePixelType>
class CudaProcessBuffer;

template<typename ImagePixelType>
class CudaStorageBuffer
{
public:
	CudaStorageBuffer(const ImagePixelType* image, Vec<unsigned int> dims, int device=0)
	{
		defaults();
		imageDims = dims;
		this->device = device;
		deviceSetup();
		memoryAllocation();
		setImage(image,cudaMemcpyHostToDevice);
		
	}

	CudaStorageBuffer(const CudaProcessBuffer<ImagePixelType>* imageBuff)
	{
		defaults();
		imageDims = imageBuff->getDimension();
		device = imageBuff->getDevice();
		deviceSetup();
		memoryAllocation();
		setImage(imageBuff->getCudaBuffer(),cudaMemcpyDeviceToDevice);
	}

	~CudaStorageBuffer()
	{
		if (deviceImage!=NULL)
			HANDLE_ERROR(cudaFree(deviceImage));

		defaults();
	}

	CudaStorageBuffer(const CudaStorageBuffer<ImagePixelType>* bufferIn){copy(bufferIn);}

	CudaStorageBuffer& operator=(const CudaStorageBuffer<ImagePixelType>* bufferIn)
	{
		if (this == bufferIn)
			return *this;

		copy(bufferIn);
		return *this;
	}

	Vec<unsigned int> getDims() const {return imageDims;}
	int getDevice() const {return device;}
	const ImagePixelType* getImagePointer(){return deviceImage;}
	size_t getGlobalMemoryAvailable() {return deviceProp.totalGlobalMem;}

	void getRoi(ImagePixelType* roi, Vec<unsigned int> starts, Vec<unsigned int> sizes) const
	{
		cudaGetROI<<<blocks,threads>>>(deviceImage,roi,imageDims,starts,sizes);
	}

private:
	CudaStorageBuffer();

	void defaults()
	{
		imageDims = Vec<unsigned int>(0,0,0);
		device = 0;
		deviceImage = NULL;
	}

	void deviceSetup() 
	{
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
		calcBlockThread(imageDims,deviceProp,blocks,threads);
	}

	void memoryAllocation()
	{
		HANDLE_ERROR(cudaMalloc((void**)&deviceImage,sizeof(ImagePixelType)*imageDims.product()));
	}

	void setImage(const ImagePixelType* image, cudaMemcpyKind direction)
	{
		HANDLE_ERROR(cudaMemcpy(deviceImage,image,sizeof(ImagePixelType)*imageDims.product(),direction));
	}

	void copy(const CudaStorageBuffer<ImagePixelType>* buff)
	{
		defaults();
		imageDims = buff->getDims();
		device = buff->getDevice();
		deviceSetup();
		memoryAllocation();
		setImage(buff->deviceImage,cudaMemcpyDeviceToDevice);
	}

	Vec<unsigned int> imageDims;
	int device;
	cudaDeviceProp deviceProp;
	dim3 blocks, threads;
	ImagePixelType* deviceImage;
};