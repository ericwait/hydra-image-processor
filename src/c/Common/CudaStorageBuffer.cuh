#pragma once

#include "CudaUtilities.cuh"
#include "CudaKernels.cuh"
#include "CudaImageContainer.cuh"
#include "Vec.h"

template<typename ImagePixelType>
class CudaProcessBuffer;

template<typename ImagePixelType>
class CudaStorageBuffer
{
public:
	CudaStorageBuffer(const ImagePixelType* image, Vec<size_t> dims, int device=0)
	{
		defaults();
		imageDims = dims;
		this->device = device;
		deviceSetup();
		memoryAllocation();
		setImage(image);
	}

// 	CudaStorageBuffer(const CudaProcessBuffer<ImagePixelType>* imageBuff)
// 	{
// 		defaults();
// 		imageDims = imageBuff->getDimension();
// 		device = imageBuff->getDevice();
// 		deviceSetup();
// 		memoryAllocation();
// 		setImage(imageBuff->getCudaBuffer(),cudaMemcpyDeviceToDevice);
// 	}

	~CudaStorageBuffer()
	{
		if (deviceImage!=NULL)
			delete deviceImage;

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

	Vec<size_t> getDims() const {return imageDims;}
	int getDevice() const {return device;}
	const CudaImageContainer* getImageContainer() const {return deviceImage;}
	size_t getGlobalMemoryAvailable() {return deviceProp.totalGlobalMem;}

	void getRoi(ImagePixelType* roi, Vec<size_t> starts, Vec<size_t> sizes) const
	{
		cudaGetROI<<<blocks,threads>>>(*deviceImage,roi,starts,sizes);
	}

private:
	CudaStorageBuffer();

	void defaults()
	{
		imageDims = Vec<size_t>(0,0,0);
		device = 0;
		deviceImage = NULL;
	}

	void deviceSetup() 
	{
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
		calcBlockThread(imageDims,deviceProp,blocks,threads);
	}

	void memoryAllocation(bool isColumnMajor=false)
	{
		deviceImage = new CudaImageContainer(imageDims,device,isColumnMajor);
	}

	void setImage(const ImagePixelType* image)
	{
		deviceImage->loadImage(image,imageDims);
	}

	void copy(const CudaStorageBuffer<ImagePixelType>* buff)
	{
		defaults();
		imageDims = buff->getDims();
		device = buff->getDevice();
		deviceSetup();
		deviceImage = new CudaImageContainer(*buff->getImageContainer());
	}

	Vec<size_t> imageDims;
	int device;
	cudaDeviceProp deviceProp;
	dim3 blocks, threads;
	CudaImageContainer* deviceImage;
};