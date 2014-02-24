#pragma once
#include "Vec.h"
#include "CudaImageContainer.cuh"

class ImageChunk
{
public:
	Vec<size_t> getFullChunkSize(){return imageEnd - imageStart;}
	Vec<size_t> getROIsize(){return chunkROIend - chunkROIstart;}

	bool sendROI(const DevicePixelType* orgImage, Vec<size_t>dims, CudaImageContainer* deviceImage);
	void retriveROI(DevicePixelType* outImage, Vec<size_t>dims, const CudaImageContainer* deviceImage);

	Vec<size_t> imageStart;
	Vec<size_t> chunkROIstart;
	Vec<size_t> imageROIstart;
	Vec<size_t> imageEnd;
	Vec<size_t> chunkROIend;
	Vec<size_t> imageROIend;

	dim3 blocks, threads;
};

