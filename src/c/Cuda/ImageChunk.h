#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "ImageContainer.h"

#include <cuda_runtime.h>
#include <vector>

class ImageChunk
{
public:
	Vec<size_t> getFullChunkSize(){return imageEnd - imageStart + 1;}
	Vec<size_t> getROIsize(){return chunkROIend - chunkROIstart + 1;}

	template <typename PixelType>
	bool sendROI(const PixelType* imageIn, Vec<size_t> dims, CudaImageContainer<PixelType>* deviceImage)
	{
		if (!deviceImage->setDims(getFullChunkSize()))
			return false;

		if (getFullChunkSize()>=dims)
		{
			deviceImage->loadImage(imageIn,dims);
			return true;
		}

		Vec<size_t> curIdx(0,0,0);
		for (curIdx.z=0; curIdx.z<getFullChunkSize().z; ++curIdx.z)
		{
			for (curIdx.y=0; curIdx.y<getFullChunkSize().y; ++curIdx.y)
			{
				Vec<size_t> curHostIdx = imageStart + curIdx;
				Vec<size_t> curDeviceIdx = curIdx;

				const PixelType* hostPtr = imageIn + dims.linearAddressAt(curHostIdx);
				PixelType* buffPtr = deviceImage->getDeviceImagePointer() + getFullChunkSize().linearAddressAt(curDeviceIdx);

				HANDLE_ERROR(cudaMemcpy(buffPtr,hostPtr,sizeof(PixelType)*getFullChunkSize().x,cudaMemcpyHostToDevice));
			}
		}
		return true;
	}

	template <typename PixelTypeIn, typename PixelTypeOut>
	bool sendROI(const PixelTypeIn* imageIn, Vec<size_t> dims, CudaImageContainer<PixelTypeOut>* deviceImage)
	{
		if (!deviceImage->setDims(getFullChunkSize()))
			return false;

		PixelTypeOut* tempBuffer = new PixelTypeOut[getFullChunkSize().x];

		Vec<size_t> curIdx(0,0,0);
		for (curIdx.z=0; curIdx.z<getFullChunkSize().z; ++curIdx.z)
		{
			for (curIdx.y=0; curIdx.y<getFullChunkSize().y; ++curIdx.y)
			{
				Vec<size_t> curHostIdx = imageStart + curIdx;
				Vec<size_t> curDeviceIdx = curIdx;

				const PixelTypeIn* hostPtr = imageIn + dims.linearAddressAt(curHostIdx);
				PixelTypeOut* buffPtr = deviceImage->getDeviceImagePointer() + getFullChunkSize().linearAddressAt(curDeviceIdx);

				for (size_t i=0; i<getFullChunkSize().x; ++i)
					tempBuffer[i] = (PixelTypeOut)(hostPtr[i]);

				HANDLE_ERROR(cudaMemcpy(buffPtr,tempBuffer,sizeof(PixelTypeOut)*getFullChunkSize().x,cudaMemcpyHostToDevice));
			}
		}

		delete[] tempBuffer;
		return true;
	}

	template <typename PixelType>
	void retriveROI(PixelType* outImage, Vec<size_t>dims, const CudaImageContainer<PixelType>* deviceImage)
	{
		cudaThreadSynchronize(); 
		gpuErrchk(cudaPeekAtLastError());

		if (getFullChunkSize()==dims)
		{
			HANDLE_ERROR(cudaMemcpy(outImage, deviceImage->getConstImagePointer(), sizeof(PixelType)*getFullChunkSize().product(),
				cudaMemcpyDeviceToHost));
		}
		else
		{
			Vec<size_t> roiIdx = Vec<size_t>(0,0,0);
			for (roiIdx.z=0; roiIdx.z<getROIsize().z; ++roiIdx.z)
			{
				for (roiIdx.y=0; roiIdx.y<getROIsize().y; ++roiIdx.y)
				{
					Vec<size_t> chunkIdx(chunkROIstart+roiIdx);
					Vec<size_t> outIdx(imageROIstart+roiIdx);

					PixelType* outPtr = outImage + dims.linearAddressAt(outIdx);
					const PixelType* chunkPtr = deviceImage->getConstImagePointer() + getFullChunkSize().linearAddressAt(chunkIdx);

					HANDLE_ERROR(cudaMemcpy(outPtr,chunkPtr,sizeof(PixelType)*getROIsize().x,cudaMemcpyDeviceToHost));
				}
			}
		}
	}

	template <typename PixelTypeIn, typename PixelTypeOut>
	void retriveROI(PixelTypeOut* outImage, Vec<size_t>dims, const CudaImageContainer<PixelTypeIn>* deviceImage)
	{
		cudaThreadSynchronize();
		gpuErrchk(cudaPeekAtLastError());

		PixelTypeIn* tempBuffer;
		if (getFullChunkSize()==dims)
		{
			tempBuffer = new PixelTypeIn[getFullChunkSize().product()];
			HANDLE_ERROR(cudaMemcpy(tempBuffer, deviceImage->getConstImagePointer(), sizeof(PixelTypeIn)*getFullChunkSize().product(),
				cudaMemcpyDeviceToHost));

			for (size_t i=0; i<getFullChunkSize().product(); ++i)
				outImage[i] = (PixelTypeOut)(tempBuffer[i]);
		}
		else
		{
			tempBuffer = new PixelTypeIn[getROIsize().x];
			Vec<size_t> roiIdx = Vec<size_t>(0,0,0);
			for (roiIdx.z=0; roiIdx.z<getROIsize().z; ++roiIdx.z)
			{
				for (roiIdx.y=0; roiIdx.y<getROIsize().y; ++roiIdx.y)
				{
					Vec<size_t> chunkIdx(chunkROIstart+roiIdx);
					Vec<size_t> outIdx(imageROIstart+roiIdx);

					PixelTypeOut* outPtr = outImage + dims.linearAddressAt(outIdx);
					const PixelTypeIn* chunkPtr = deviceImage->getConstImagePointer() + deviceImage->getDims().linearAddressAt(chunkIdx);

					HANDLE_ERROR(cudaMemcpy(tempBuffer,chunkPtr,sizeof(PixelTypeIn)*getROIsize().x,cudaMemcpyDeviceToHost));

					for (size_t i=0; i<getROIsize().x; ++i)
						outPtr[i] = (PixelTypeOut)tempBuffer[i];
				}
			}
		}
		delete[] tempBuffer;
	}

    // This chunk starts at this location in the original image
	Vec<size_t> imageStart;

    // This is the start of the chunk to be evaluated
	Vec<size_t> chunkROIstart;

    // This is where the chunk should go back to the original image
	Vec<size_t> imageROIstart;

	size_t channelStart;

	size_t frameStart;

    // This chunk ends at this location in the original image (inclusive)
	Vec<size_t> imageEnd;

    // This is the end of the chunk to be evaluated (inclusive)
	Vec<size_t> chunkROIend;

    // This is where the chunk should go back to the original image (inclusive)
	Vec<size_t> imageROIend;

	size_t channelEnd;

	size_t frameEnd;

	dim3 blocks, threads;
};

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims);

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, Vec<size_t> deviceDims, size_t maxThreads, Vec<size_t> kernalDims=Vec<size_t>(0,0,0));

std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, size_t memAvailable, size_t bytesPerVal, size_t maxThreads, Vec<size_t> kernelDims=Vec<size_t>(0,0,0));
