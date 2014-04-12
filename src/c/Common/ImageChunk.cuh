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

