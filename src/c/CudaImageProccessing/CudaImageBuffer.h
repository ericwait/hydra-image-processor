#ifndef CUDA_IMAGE_BUFFER_H
#define CUDA_IMAGE_BUFFER_H

#include "Vec.h"

#define NUM_BUFFERS (2)

template<typename ImagePixelType>
class CudaImageBuffer
{
public:
	CudaImageBuffer(Vec<int> dims, int device=0)
	{
		this->dims = dims;
		this->device = device;

		MemoryAllocation();
	}

	CudaImageBuffer(int x, int y, int z, int device=0)
	{
		dims(x,y,z);
		this->device = device;

		MemoryAllocation();
	}

	CudaImageBuffer(int x, int y, int device=0)
	{
		dims(x,y,1);
		this->device = device;

		MemoryAllocation();
	}

	CudaImageBuffer(int x, int device=0)
	{
		dims(x,1,1);
		this->device = device;

		MemoryAllocation();
	}

	~CudaImageBuffer()
	{
		HANDLE_ERROR(cudaFree(imageBuffer0));
		HANDLE_ERROR(cudaFree(imageBuffer1));
	}

	CudaImageBuffer (const CudaImageBuffer& bufferIn){copy(bufferIn);}
	CudaImageBuffer operator=(const CudaImageBuffer& bufferIn){copy(bufferIn);}

	void loadImage(ImagePixelType* image)
	{
		incrementBufferNumber();
		HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],image,sizeof(ImagePixelType)*dims.product(),cudaMemcpyHostToDevice));
	}

	void retrieveImage(ImagePixelType& imageOut)
	{
		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
			return NULL;

		HANDLE_ERROR(cudaMemcpy(image,imageBuffers[currentBuffer],sizeof(ImagePixelType)*dims.product(),cudaMemcpyDeviceToHost));
	}

	Vec<int> getDimension(){return dims;}
	int getDevice(){return device;}

private:
	CudaImageBuffer();
	void MemoryAllocation()
	{
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaMalloc((void**)&imageBuffer0,sizeof(ImagePixelType)*dims.product()));
		HANDLE_ERROR(cudaMalloc((void**)&imageBuffer1,sizeof(ImagePixelType)*dims.product()));

		currentBuffer = -1;
	}

	void copy(const CudaImageBuffer bufferIn)
	{
		dims = bufferIn.getDimension();
		device = bufferIn.getDevice();

		MemoryAllocation();

		currentBuffer = 0;
		ImagePixelType* inImage = bufferIn.getCurrentBuffer();

		if (inImage!=NULL)
			HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],inImage,sizeof(ImagePixelType)*dims.product(),cudaMemcpyHostToHost));
	}

	ImagePixelType* getCurrentBuffer()
	{
		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
			return NULL;

		return imageBuffers[currentBuffer];
	}

	void incrementBufferNumber()
	{
		++currentBuffer;

		if (currentBuffer>=NUM_BUFFERS)
			currentBuffer = 0;
	}

	Vec<int> dims;
	int device;
	char currentBuffer;
	ImagePixelType* imageBuffers[NUM_BUFFERS];
};
#endif