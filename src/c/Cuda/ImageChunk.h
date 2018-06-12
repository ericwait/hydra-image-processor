#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "ImageDimensions.cuh"
#include "CudaDeviceInfo.h"
#include "BufferConversions.h"

#include <cuda_runtime.h>
#include <vector>

class ImageChunk
{
public:
	Vec<size_t> getFullChunkSize();
	Vec<size_t> getROIsize() { return chunkROIend-chunkROIstart+1; }

	template <typename PixelTypeOut, typename PixelTypeIn>
	void copyLine(PixelTypeOut* dst, PixelTypeIn* src, size_t length, cudaMemcpyKind direction)
	{
		if (direction == cudaMemcpyHostToDevice)
		{
			PixelTypeOut* tempBuffer;
			toDevice(&tempBuffer, src, length);
			HANDLE_ERROR(cudaMemcpy(dst, tempBuffer, sizeof(PixelTypeOut)*length, direction));
			cleanBuffer(&tempBuffer, src);
		}
		else if(direction == cudaMemcpyDeviceToHost)
		{
			PixelTypeIn* tempBuffer;
			fromDevice(&tempBuffer, &dst, length);
			HANDLE_ERROR(cudaMemcpy(tempBuffer, src, sizeof(PixelTypeIn)*length, direction));
			copyBuffer(&dst, &tempBuffer, length);
		}
		else
		{
			std::runtime_error("This copy direction is not supported!");
		}
	}

	template <typename PixelTypeIn, typename PixelTypeOut>
	bool sendROI(ImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut>* deviceImage)
	{
		Vec<size_t> chunkVolSize = getFullChunkSize();
		Vec<size_t> imVolSize = imageIn.getSpatialDims();
		
		if (!deviceImage->setDims(chunkVolSize))
			return false;

		if (chunkVolSize == imVolSize)
		{
			ImageDimensions curHostCoord(imageStart, channel,frame);

			size_t hostOffset = imageIn.getDims().linearAddressAt(curHostCoord);
			PixelTypeIn* hostPtr = imageIn.getPtr() + hostOffset;
			PixelTypeOut* buffPtr = deviceImage->getDeviceImagePointer();

			copyLine(buffPtr, hostPtr, chunkVolSize.product(), cudaMemcpyHostToDevice);
			return true;

		}

		Vec<size_t> curPos(0);
		for (curPos.z = 0; curPos.z < chunkVolSize.z; ++curPos.z)
		{
			for (curPos.y = 0; curPos.y < chunkVolSize.y; ++curPos.y)
			{
				ImageDimensions curHostCoord(imageStart+curPos, channel, frame);
				Vec<size_t> curDeviceCoord = curPos;

				size_t hostOffset = imageIn.getDims().linearAddressAt(curHostCoord);
				PixelTypeIn* hostPtr = imageIn.getPtr() + hostOffset;

				size_t deviceOffset = deviceImage->getDims().linearAddressAt(curDeviceCoord);
				PixelTypeOut* buffPtr = deviceImage->getDeviceImagePointer() + deviceOffset;

				copyLine(buffPtr, hostPtr, chunkVolSize.x, cudaMemcpyHostToDevice);
			}
		}

		return true;
	}

	template <typename PixelTypeOut, typename PixelTypeIn>
	void retriveROI(ImageContainer<PixelTypeOut> outImage, CudaImageContainer<PixelTypeIn>* deviceImage)
	{
		cudaThreadSynchronize();
		HANDLE_ERROR(cudaPeekAtLastError());

		Vec<size_t> chunkVolSize = getFullChunkSize();
		Vec<size_t> imVolSize = outImage.getSpatialDims();

		if (chunkVolSize == imVolSize)
		{
			ImageDimensions curHostCoord(imageStart,channel,frame);

			size_t hostOffset = outImage.getDims().linearAddressAt(curHostCoord);
			PixelTypeOut* hostPtr = outImage.getPtr() + hostOffset;
			PixelTypeIn* buffPtr = deviceImage->getDeviceImagePointer();

			copyLine(hostPtr, buffPtr, chunkVolSize.product(), cudaMemcpyDeviceToHost);
			return;
		}

		Vec<size_t> curPos(0);
		for (curPos.z = 0; curPos.z < chunkVolSize.z; ++curPos.z)
		{
			for (curPos.y = 0; curPos.y < chunkVolSize.y; ++curPos.y)
			{
				ImageDimensions curHostCoord(imageStart + curPos, channel, frame);
				Vec<size_t> curDeviceCoord = curPos;

				size_t hostOffset = outImage.getDims().linearAddressAt(curHostCoord);
				PixelTypeOut* hostPtr = outImage.getPtr() + hostOffset;

				size_t deviceOffset = deviceImage->getDims().linearAddressAt(curDeviceCoord);
				PixelTypeIn* buffPtr = deviceImage->getDeviceImagePointer() + deviceOffset;

				copyLine(hostPtr, buffPtr, chunkVolSize.x, cudaMemcpyDeviceToHost);
			}
		}
	}

	// This chunk starts at this location in the original image
	Vec<size_t> imageStart;

	// This is the start of the chunk to be evaluated
	Vec<size_t> chunkROIstart;

	// This is where the chunk should go back to the original image
	Vec<size_t> imageROIstart;

	// This chunk ends at this location in the original image (inclusive)
	Vec<size_t> imageEnd;

	// This is the end of the chunk to be evaluated (inclusive)
	Vec<size_t> chunkROIend;

	// This is where the chunk should go back to the original image (inclusive)
	Vec<size_t> imageROIend;

	// This is the channel that this chunk will operate on.
	size_t channel;

	// This is the frame that this chunk will operate on.
	size_t frame;

	dim3 blocks, threads;
};

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims);

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, Vec<size_t> deviceDims, CudaDevices maxThreads, Vec<size_t> kernalDims = Vec<size_t>(1, 1, 1));

std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, CudaDevices cudaDevs, size_t bytesPerVal, Vec<size_t> kernelDims = Vec<size_t>(1, 1, 1), float memMultiplier=1.0f);
