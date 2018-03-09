#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"
#include "ImageDimensions.cuh"
#include "CudaDeviceInfo.h"

#include <cuda_runtime.h>
#include <vector>

class ImageChunk
{
public:
	ImageDimensions getFullChunkSize() { return ImageDimensions(imageEnd-imageStart+1,channelEnd-channelStart+1,frameEnd-frameStart+1); }
	Vec<size_t> getROIsize() { return chunkROIend-chunkROIstart+1; }

	template <typename PixelType>
	bool sendROI(const ImageContainer<PixelType> imageIn, CudaImageContainer<PixelType>* deviceImage)
	{
		if(!deviceImage->setDims(getFullChunkSize()))
			return false;

		ImageDimensions stopPos(getFullChunkSize());
		if(stopPos>=imageIn.getDims())
		{
			deviceImage->loadImage(imageIn);
			return true;
		}

		ImageDimensions curPos(0, 0, 0);
		for(curPos.frame = frameStart; curPos.frame<stopPos.frame; ++curPos.frame)
		{
			for(curPos.chan = channelStart; curPos.chan<stopPos.chan; ++curPos.chan)
			{
				for(curPos.dims.z = 0; curPos.dims.z<stopPos.dims.z; ++curPos.dims.z)
				{
					for(curPos.dims.y = 0; curPos.dims.y<stopPos.dims.y; ++curPos.dims.y)
					{
						ImageDimensions curHostIdx(imageStart+curPos.dims, channelStart+curPos.chan, frameStart+curPos.frame);
						ImageDimensions curDeviceIdx = curPos;

						size_t hostOffset = imageIn.getDims().linearAddressAt(curHostIdx);
						const PixelType* hostPtr = imageIn.getPtr() +hostOffset;

						size_t deviceOffset = deviceImage->getDims().linearAddressAt(curDeviceIdx);
						void* buffPtr = deviceImage->getDeviceImagePointer() + deviceOffset;

						HANDLE_ERROR(cudaMemcpy(buffPtr, hostPtr, sizeof(PixelType)*stopPos.dims.x, cudaMemcpyHostToDevice));
					}
				}
			}
		}
		return true;
	}

	template <typename PixelTypeIn, typename PixelTypeOut>
	bool sendROI(const ImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut>* deviceImage)
	{
		DEBUG_KERNEL_CHECK();
		if(!deviceImage->setDims(getFullChunkSize()))
			return false;

		PixelTypeOut* tempBuffer = new PixelTypeOut[getFullChunkSize().dims.x];

		ImageDimensions curPos(0, 0, 0);
		ImageDimensions stopPos(getFullChunkSize());
		for(curPos.frame = frameStart; curPos.frame<stopPos.frame; ++curPos.frame)
		{
			for(curPos.chan = channelStart; curPos.chan<stopPos.chan; ++curPos.chan)
			{
				for(curPos.dims.z = 0; curPos.dims.z<stopPos.dims.z; ++curPos.dims.z)
				{
					for(curPos.dims.y = 0; curPos.dims.y<stopPos.dims.y; ++curPos.dims.y)
					{
						ImageDimensions curHostIdx(imageStart+curPos.dims, channelStart+curPos.chan, frameStart+curPos.frame);
						ImageDimensions curDeviceIdx = curPos;

						const PixelTypeIn* hostPtr = imageIn.getConstPtr() + imageIn.getDims().linearAddressAt(curHostIdx);
						PixelTypeOut* buffPtr = deviceImage->getDeviceImagePointer() + deviceImage->getDims().linearAddressAt(curDeviceIdx);

						for(size_t i = 0; i<stopPos.dims.x; ++i)
							tempBuffer[i] = (PixelTypeOut)(hostPtr[i]);

						HANDLE_ERROR(cudaMemcpy(buffPtr, tempBuffer, sizeof(PixelTypeOut)*stopPos.dims.x, cudaMemcpyHostToDevice));
					}
				}
			}
		}

		delete[] tempBuffer;
		return true;
	}

	template <typename PixelType>
	void retriveROI(ImageContainer<PixelType> outImage, const CudaImageContainer<PixelType>* deviceImage)
	{
		cudaThreadSynchronize();
		GPU_ERROR_CHK(cudaPeekAtLastError());

		if(getFullChunkSize()==outImage.getDims())
		{
			HANDLE_ERROR(cudaMemcpy(outImage.getPtr(), deviceImage->getConstImagePointer(), sizeof(PixelType)*getFullChunkSize().getNumElements(), cudaMemcpyDeviceToHost));
			return;
		} 

		ImageDimensions curPos(0, 0, 0);
		ImageDimensions stopPos(getFullChunkSize());
		for(curPos.frame = frameStart; curPos.frame<stopPos.frame; ++curPos.frame)
		{
			for(curPos.chan = channelStart; curPos.chan<stopPos.chan; ++curPos.chan)
			{
				for(curPos.dims.z = 0; curPos.dims.z<stopPos.dims.z; ++curPos.dims.z)
				{
					for(curPos.dims.y = 0; curPos.dims.y<stopPos.dims.y; ++curPos.dims.y)
					{
						ImageDimensions curHostIdx(imageStart+curPos.dims, channelStart+curPos.chan, frameStart+curPos.frame);
						ImageDimensions curDeviceIdx = curPos;

						size_t hostOffset = outImage.getDims().linearAddressAt(curHostIdx);
						PixelType* hostPtr = outImage.getPtr() + hostOffset;

						size_t deviceOffset = deviceImage->getDims().linearAddressAt(curDeviceIdx);
						const PixelType* buffPtr = deviceImage->getDeviceImagePointer() + deviceOffset;

						HANDLE_ERROR(cudaMemcpy(hostPtr, buffPtr, sizeof(PixelType)*stopPos.dims.x, cudaMemcpyHostToDevice));
					}
				}
			}
		}
	}

	template <typename PixelTypeIn, typename PixelTypeOut>
	void retriveROI(ImageContainer<PixelTypeOut> outImage, const CudaImageContainer<PixelTypeIn>* deviceImage)
	{
		cudaThreadSynchronize();
		HANDLE_ERROR(cudaPeekAtLastError());

		PixelTypeIn* tempBuffer;
		if(getFullChunkSize()==outImage.dims)
		{
			size_t numVals = getFullChunkSize().getNumElements();
			tempBuffer = new PixelTypeIn[numVals];
			HANDLE_ERROR(cudaMemcpy(tempBuffer, deviceImage->getConstImagePointer(), sizeof(PixelTypeIn)*numVals, cudaMemcpyDeviceToHost));

			for(size_t i = 0; i<numVals; ++i)
				outImage.getPtr()[i] = (PixelTypeOut)(tempBuffer[i]);
		}
		else
		{
			ImageDimensions curPos(0, 0, 0);
			ImageDimensions stopPos(getFullChunkSize());
			tempBuffer = new PixelTypeIn[stopPos.dims.x];
			for(curPos.frame = frameStart; curPos.frame<stopPos.frame; ++curPos.frame)
			{
				for(curPos.chan = channelStart; curPos.chan<stopPos.chan; ++curPos.chan)
				{
					for(curPos.dims.z = 0; curPos.dims.z<stopPos.dims.z; ++curPos.dims.z)
					{
						for(curPos.dims.y = 0; curPos.dims.y<stopPos.dims.y; ++curPos.dims.y)
						{
							ImageDimensions curHostIdx(imageStart+curPos.dims, channelStart+curPos.chan, frameStart+curPos.frame);
							ImageDimensions curDeviceIdx = curPos;

							PixelTypeOut* hostPtr = outImage.getPtr() + outImage.getDims().linearAddressAt(curHostIdx);
							const PixelTypeIn* buffPtr = deviceImage->getDeviceImagePointer()+deviceImage->getDims().linearAddressAt(curDeviceIdx);

							HANDLE_ERROR(cudaMemcpy(tempBuffer, buffPtr, sizeof(PixelTypeIn)*stopPos.dims.x, cudaMemcpyHostToDevice));
							
							for (size_t i = 0; i < stopPos.dims.x; ++i)
							{

								hostPtr[i] = (PixelTypeOut)(tempBuffer[i]);
							}
						}
					}
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

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, ImageDimensions &maxDeviceDims);

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, ImageDimensions deviceDims, size_t maxThreads, Vec<size_t> kernalDims = Vec<size_t>(0, 0, 0));

std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, CudaDevices cudaDevs, size_t bytesPerVal, Vec<size_t> kernelDims = Vec<size_t>(0, 0, 0));
