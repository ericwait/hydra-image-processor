#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "ImageDimensions.cuh"
#include "CudaDeviceInfo.h"

#include <cuda_runtime.h>
#include <vector>
#include <type_traits>

class ImageChunk
{
public:
	ImageDimensions getFullChunkSize();
	Vec<size_t> getROIsize() { return chunkROIend-chunkROIstart+1; }

	template <typename PixelTypeOut, typename PixelTypeIn>
	void copyLine(PixelTypeOut* dst, PixelTypeIn* src, size_t length, cudaMemcpyKind direction)
	{
		if (direction == cudaMemcpyHostToDevice)
		{
			PixelTypeOut* tempBuffer;
			if (std::is_same<PixelTypeIn,PixelTypeOut>::value)
			{
				tempBuffer = src;
			}
			else
			{
				tempBuffer = new PixelTypeOut[length];
				for (size_t i = 0; i < length; ++i)
					tempBuffer[i] = (PixelTypeOut)(src[i]);
			}

			HANDLE_ERROR(cudaMemcpy(dst, tempBuffer, sizeof(PixelTypeOut)*length, direction));

			if (!std::is_same<PixelTypeIn, PixelTypeOut>::value)
			{
				delete[] tempBuffer;
			}
		}
		else if(direction == cudaMemcpyDeviceToHost)
		{
			PixelTypeIn* tempBuffer;
			if (std::is_same<PixelTypeIn, PixelTypeOut>::value)
			{
				tempBuffer = dst;
			}
			else
			{
				tempBuffer = new PixelTypeIn[length];
			}

			HANDLE_ERROR(cudaMemcpy(tempBuffer, src, sizeof(PixelTypeIn)*length, direction));

			if (!std::is_same<PixelTypeIn, PixelTypeOut>::value)
			{
				for (size_t i = 0; i < length; ++i)
					dst[i] = (PixelTypeOut)(tempBuffer[i]);

				delete[] tempBuffer;
			}
		}
		else
		{
			std::runtime_error("This copy direction is not supported!");
		}
	}

	template <typename PixelTypeIn, typename PixelTypeOut>
	bool sendROI(ImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut>* deviceImage)
	{
		if(!deviceImage->setDims(getFullChunkSize().dims))
			return false;

		ImageDimensions stopPos(getFullChunkSize());
		if(stopPos>=imageIn.getDims() && std::is_same<PixelTypeIn,PixelTypeOut>::value)
		{
			deviceImage->loadImage(imageIn.getPtr(),imageIn.getNumElements());
			return true;
		}

		ImageDimensions curPos(0, 0, 0);
		for(curPos.frame = 0; curPos.frame<stopPos.frame; ++curPos.frame)
		{
			for(curPos.chan = 0; curPos.chan<stopPos.chan; ++curPos.chan)
			{
				for(curPos.dims.z = 0; curPos.dims.z<stopPos.dims.z; ++curPos.dims.z)
				{
					for(curPos.dims.y = 0; curPos.dims.y<stopPos.dims.y; ++curPos.dims.y)
					{
						ImageDimensions curHostIdx(imageStart+curPos);
						ImageDimensions curDeviceIdx = curPos;

						size_t hostOffset = imageIn.getDims().linearAddressAt(curHostIdx);
						PixelTypeIn* hostPtr = imageIn.getPtr() +hostOffset;

						size_t deviceOffset = deviceImage->getDims().linearAddressAt(curDeviceIdx.dims);
						PixelTypeOut* buffPtr = deviceImage->getDeviceImagePointer() + deviceOffset;

						copyLine(buffPtr, hostPtr, stopPos.dims.x, cudaMemcpyHostToDevice);
					}
				}
			}
		}
		return true;
	}

	template <typename PixelTypeOut, typename PixelTypeIn>
	void retriveROI(ImageContainer<PixelTypeOut> outImage, CudaImageContainer<PixelTypeIn>* deviceImage)
	{
		cudaThreadSynchronize();
		GPU_ERROR_CHK(cudaPeekAtLastError());

		if(getFullChunkSize()==outImage.getDims())
		{
			copyLine(outImage.getPtr(), deviceImage->getImagePointer(), getFullChunkSize().getNumElements(), cudaMemcpyDeviceToHost);
			return;
		} 

		ImageDimensions curPos(0, 0, 0);
		ImageDimensions stopPos(getFullChunkSize());
		for(curPos.frame = 0; curPos.frame<stopPos.frame; ++curPos.frame)
		{
			for(curPos.chan = 0; curPos.chan<stopPos.chan; ++curPos.chan)
			{
				for(curPos.dims.z = 0; curPos.dims.z<stopPos.dims.z; ++curPos.dims.z)
				{
					for(curPos.dims.y = 0; curPos.dims.y<stopPos.dims.y; ++curPos.dims.y)
					{
						ImageDimensions curHostIdx(imageStart+curPos);
						ImageDimensions curDeviceIdx = curPos;

						size_t hostOffset = outImage.getDims().linearAddressAt(curHostIdx);
						PixelTypeOut* hostPtr = outImage.getPtr() + hostOffset;

						size_t deviceOffset = deviceImage->getDims().linearAddressAt(curDeviceIdx.dims);
						PixelTypeIn* buffPtr = deviceImage->getDeviceImagePointer() + deviceOffset;

						copyLine(hostPtr, buffPtr, stopPos.dims.x, cudaMemcpyDeviceToHost);
					}
				}
			}
		}
	}

	// This chunk starts at this location in the original image
	ImageDimensions imageStart;

	// This is the start of the chunk to be evaluated
	Vec<size_t> chunkROIstart;

	// This is where the chunk should go back to the original image
	Vec<size_t> imageROIstart;

	size_t channelStart;

	size_t frameStart;

	// This chunk ends at this location in the original image (inclusive)
	ImageDimensions imageEnd;

	// This is the end of the chunk to be evaluated (inclusive)
	Vec<size_t> chunkROIend;

	// This is where the chunk should go back to the original image (inclusive)
	Vec<size_t> imageROIend;

	size_t channelEnd;

	size_t frameEnd;

	dim3 blocks, threads;
};

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims);

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, Vec<size_t> deviceDims, CudaDevices maxThreads, Vec<size_t> kernalDims = Vec<size_t>(1, 1, 1));

std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, CudaDevices cudaDevs, size_t bytesPerVal, Vec<size_t> kernelDims = Vec<size_t>(1, 1, 1));
