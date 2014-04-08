#include "ImageChunk.cuh"
#include "cuda_runtime.h"

bool ImageChunk::sendROI(const DevicePixelType* imageIn, Vec<size_t>orgImageDims, CudaImageContainer<DevicePixelType>* deviceImage)
{
	if (!deviceImage->setDims(getFullChunkSize()))
		return false;

	if (getFullChunkSize()>=orgImageDims)
	{
		deviceImage->loadImage(imageIn,orgImageDims);
		return true;
	}

	Vec<size_t> curIdx(0,0,0);
	for (curIdx.z=0; curIdx.z<getFullChunkSize().z; ++curIdx.z)
	{
		for (curIdx.y=0; curIdx.y<getFullChunkSize().y; ++curIdx.y)
		{
			Vec<size_t> curHostIdx = imageStart + curIdx;
			Vec<size_t> curDeviceIdx = curIdx;

			const DevicePixelType* hostPtr = imageIn + orgImageDims.linearAddressAt(curHostIdx);
			DevicePixelType* buffPtr = deviceImage->getDeviceImagePointer() + getFullChunkSize().linearAddressAt(curDeviceIdx);

			HANDLE_ERROR(cudaMemcpy(buffPtr,hostPtr,sizeof(DevicePixelType)*getFullChunkSize().x,cudaMemcpyHostToDevice));
		}
	}
	return true;
}

void ImageChunk::retriveROI(DevicePixelType* outImage, Vec<size_t>dims, const CudaImageContainer<DevicePixelType>* deviceImage)
{
	if (getFullChunkSize()==dims)
	{
		HANDLE_ERROR(cudaMemcpy(outImage, deviceImage->getConstImagePointer(), sizeof(DevicePixelType)*getFullChunkSize().product(),
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

				DevicePixelType* outPtr = outImage + dims.linearAddressAt(outIdx);
				const DevicePixelType* chunkPtr = deviceImage->getConstImagePointer() + deviceImage->getDims().linearAddressAt(chunkIdx);

				HANDLE_ERROR(cudaMemcpy(outPtr,chunkPtr,sizeof(DevicePixelType)*getROIsize().x,cudaMemcpyDeviceToHost));
			}
		}
	}
}

