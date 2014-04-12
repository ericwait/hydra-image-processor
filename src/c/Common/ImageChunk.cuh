#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"

class ImageChunk
{
public:
	Vec<size_t> getFullChunkSize(){return imageEnd - imageStart;}
	Vec<size_t> getROIsize(){return chunkROIend - chunkROIstart;}

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

	template <typename PixelType>
	void retriveROI(PixelType* outImage, Vec<size_t>dims, const CudaImageContainer<PixelType>* deviceImage)
	{
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
					const PixelType* chunkPtr = deviceImage->getConstImagePointer() + deviceImage->getDims().linearAddressAt(chunkIdx);

					HANDLE_ERROR(cudaMemcpy(outPtr,chunkPtr,sizeof(PixelType)*getROIsize().x,cudaMemcpyDeviceToHost));
				}
			}
		}
	}

	Vec<size_t> imageStart;
	Vec<size_t> chunkROIstart;
	Vec<size_t> imageROIstart;
	Vec<size_t> imageEnd;
	Vec<size_t> chunkROIend;
	Vec<size_t> imageROIend;

	dim3 blocks, threads;
};

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims);

std::vector<ImageChunk> calculateChunking(Vec<size_t> orgImageDims, Vec<size_t> deviceDims, const cudaDeviceProp& prop,
										  Vec<size_t> kernalDims=Vec<size_t>(0,0,0));

template <class PixelType>
std::vector<ImageChunk> calculateBuffers(Vec<size_t> imageDims, int numBuffersNeeded, size_t memAvailable, const cudaDeviceProp& prop,
										 Vec<size_t> kernelDims=Vec<size_t>(0,0,0))
{
	size_t numVoxels = (size_t)(memAvailable / (sizeof(PixelType)*numBuffersNeeded));

	Vec<size_t> overlapVolume;
	overlapVolume.x = kernelDims.x * imageDims.y * imageDims.z;
	overlapVolume.y = imageDims.x * kernelDims.y * imageDims.z;
	overlapVolume.z = imageDims.x * imageDims.y * kernelDims.z;

	Vec<size_t> deviceDims(0,0,0);

	if (overlapVolume.x>overlapVolume.y && overlapVolume.x>overlapVolume.z) // chunking in X is the worst
	{
		deviceDims.x = imageDims.x;
		double leftOver = (double)numVoxels/imageDims.x;
		double squareDim = sqrt(leftOver);

		if (overlapVolume.y<overlapVolume.z) // chunking in Y is second worst
		{
			if (squareDim>imageDims.y)
				deviceDims.y = imageDims.y;
			else 
				deviceDims.y = (size_t)squareDim;

			deviceDims.z = (size_t)(leftOver/deviceDims.y);

			if (deviceDims.z>imageDims.z)
				deviceDims.z = imageDims.z;
		}
		else // chunking in Z is second worst
		{
			if (squareDim>imageDims.z)
				deviceDims.z = imageDims.z;
			else 
				deviceDims.z = (size_t)squareDim;

			deviceDims.y = (size_t)(leftOver/deviceDims.z);

			if (deviceDims.y>imageDims.y)
				deviceDims.y = imageDims.y;
		}
	}
	else if (overlapVolume.y>overlapVolume.z) // chunking in Y is the worst
	{
		deviceDims.y = imageDims.y;
		double leftOver = (double)numVoxels/imageDims.y;
		double squareDim = sqrt(leftOver);

		if (overlapVolume.x<overlapVolume.z)
		{
			if (squareDim>imageDims.x)
				deviceDims.x = imageDims.x;
			else 
				deviceDims.x = (size_t)squareDim;

			deviceDims.z = (size_t)(leftOver/deviceDims.x);

			if (deviceDims.z>imageDims.z)
				deviceDims.z = imageDims.z;
		}
		else
		{
			if (squareDim>imageDims.z)
				deviceDims.z = imageDims.z;
			else 
				deviceDims.z = (size_t)squareDim;

			deviceDims.x = (size_t)(leftOver/deviceDims.z);

			if (deviceDims.x>imageDims.x)
				deviceDims.x = imageDims.x;
		}
	}
	else // chunking in Z is the worst
	{
		deviceDims.z = imageDims.z;
		double leftOver = (double)numVoxels/imageDims.z;
		double squareDim = sqrt(leftOver);

		if (overlapVolume.x<overlapVolume.y)
		{
			if (squareDim>imageDims.x)
				deviceDims.x = imageDims.x;
			else 
				deviceDims.x = (size_t)squareDim;

			deviceDims.y = (size_t)(leftOver/deviceDims.x);

			if (deviceDims.y>imageDims.y)
				deviceDims.y = imageDims.y;
		}
		else
		{
			if (squareDim>imageDims.y)
				deviceDims.y = imageDims.y;
			else 
				deviceDims.y = (size_t)squareDim;

			deviceDims.x = (size_t)(leftOver/deviceDims.z);

			if (deviceDims.x>imageDims.x)
				deviceDims.x = imageDims.x;
		}
	}

	return calculateChunking(imageDims, deviceDims, prop, kernelDims);
}