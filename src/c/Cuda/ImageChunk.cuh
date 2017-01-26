#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"

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

    // This chunk ends at this location in the original image (inclusive)
	Vec<size_t> imageEnd;

    // This is the end of the chunk to be evaluated (inclusive)
	Vec<size_t> chunkROIend;

    // This is where the chunk should go back to the original image (inclusive)
	Vec<size_t> imageROIend;

	dim3 blocks, threads;
};

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims);

std::vector<ImageChunk> calculateChunking(Vec<size_t> orgImageDims, Vec<size_t> deviceDims, const cudaDeviceProp& prop,
										  Vec<size_t> kernalDims=Vec<size_t>(0,0,0), size_t maxThreads=std::numeric_limits<size_t>::max());

template <class PixelType>
std::vector<ImageChunk> calculateBuffers(Vec<size_t> imageDims, int numBuffersNeeded, size_t memAvailable, const cudaDeviceProp& prop,
										 Vec<size_t> kernelDims=Vec<size_t>(0,0,0), size_t maxThreads=std::numeric_limits<size_t>::max())
{
	size_t numVoxels = (size_t)(memAvailable / (sizeof(PixelType)*numBuffersNeeded));

	Vec<size_t> overlapVolume;
	overlapVolume.x = (kernelDims.x-1) * imageDims.y * imageDims.z;
	overlapVolume.y = imageDims.x * (kernelDims.y-1) * imageDims.z;
	overlapVolume.z = imageDims.x * imageDims.y * (kernelDims.z-1);

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

			deviceDims.z = (size_t)(numVoxels/(deviceDims.y*deviceDims.x));

			if (deviceDims.z>imageDims.z)
				deviceDims.z = imageDims.z;
		}
		else // chunking in Z is second worst
		{
			if (squareDim>imageDims.z)
				deviceDims.z = imageDims.z;
			else 
				deviceDims.z = (size_t)squareDim;

			deviceDims.y = (size_t)(numVoxels/(deviceDims.z*deviceDims.x));

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

			deviceDims.z = (size_t)(numVoxels/(deviceDims.x*deviceDims.y));

			if (deviceDims.z>imageDims.z)
				deviceDims.z = imageDims.z;
		}
		else
		{
			if (squareDim>imageDims.z)
				deviceDims.z = imageDims.z;
			else 
				deviceDims.z = (size_t)squareDim;

			deviceDims.x = (size_t)(numVoxels/(deviceDims.z*deviceDims.y));

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

			deviceDims.y = (size_t)(numVoxels/(deviceDims.x*deviceDims.z));

			if (deviceDims.y>imageDims.y)
				deviceDims.y = imageDims.y;
		}
		else
		{
			if (squareDim>imageDims.y)
				deviceDims.y = imageDims.y;
			else 
				deviceDims.y = (size_t)squareDim;

			deviceDims.x = (size_t)(numVoxels/(deviceDims.y*deviceDims.z));

			if (deviceDims.x>imageDims.x)
				deviceDims.x = imageDims.x;
		}
	}

	return calculateChunking(imageDims, deviceDims, prop, kernelDims,maxThreads);
}