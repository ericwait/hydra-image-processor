#include "ImageChunk.h"
#include "CudaUtilities.h"

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims)
{
	maxDeviceDims = Vec<size_t>(0,0,0);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		Vec<size_t> curDim = curChunk->getFullChunkSize();

		if (curDim.x>maxDeviceDims.x)
			maxDeviceDims.x = curDim.x;

		if (curDim.y>maxDeviceDims.y)
			maxDeviceDims.y = curDim.y;

		if (curDim.z>maxDeviceDims.z)
			maxDeviceDims.z = curDim.z;
	}
}

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, Vec<size_t> deviceDims, size_t maxThreads, Vec<size_t> kernalDims/*=Vec<size_t>(0,0,0)*/)
{
	std::vector<ImageChunk> localChunks;
	Vec<size_t> margin((kernalDims + 1)/2); //integer round
	Vec<size_t> chunkDelta(deviceDims-margin*2);
	Vec<size_t> numChunks(1,1,1);

	Vec<size_t> spatialDimensions = imageDims.spatialDimensions;

	if (spatialDimensions.x>deviceDims.x)
		numChunks.x = (size_t)ceil((double)spatialDimensions.x/chunkDelta.x);
	else
		chunkDelta.x = spatialDimensions.x;

	if (spatialDimensions.y>deviceDims.y)
		numChunks.y = (size_t)ceil((double)spatialDimensions.y/chunkDelta.y);
	else
		chunkDelta.y = spatialDimensions.y;

	if (spatialDimensions.z>deviceDims.z)
		numChunks.z = (size_t)ceil((double)spatialDimensions.z/chunkDelta.z);
	else
		chunkDelta.z = spatialDimensions.z;

	localChunks.resize(numChunks.product());

	Vec<size_t> curChunk(0,0,0);
	Vec<size_t> imageStart(0,0,0);
	Vec<size_t> chunkROIstart(0,0,0);
	Vec<size_t> imageROIstart(0,0,0);
	Vec<size_t> imageEnd(0,0,0);
	Vec<size_t> chunkROIend(0,0,0);
	Vec<size_t> imageROIend(0,0,0);

	for (curChunk.z=0; curChunk.z<numChunks.z; ++curChunk.z)
	{
		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
		{
			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
			{
				imageROIstart = chunkDelta * curChunk;
				imageROIend = Vec<size_t>::min(imageROIstart + chunkDelta, imageDims -1);
				imageStart = Vec<size_t>(Vec<int>::max(Vec<int>(imageROIstart)-Vec<int>(margin), Vec<int>(0,0,0)));
				imageEnd = Vec<size_t>::min(imageROIend + margin, imageDims-1);
				chunkROIstart = imageROIstart - imageStart;
				chunkROIend = imageROIend - imageStart;

				ImageChunk* curImageBuffer = &localChunks[numChunks.linearAddressAt(curChunk)];

				curImageBuffer->imageStart = imageStart;
				curImageBuffer->chunkROIstart = chunkROIstart;
				curImageBuffer->imageROIstart = imageROIstart;
				curImageBuffer->imageEnd = imageEnd;
				curImageBuffer->chunkROIend = chunkROIend;
				curImageBuffer->imageROIend = imageROIend;

				calcBlockThread(curImageBuffer->getFullChunkSize(),maxThreads,curImageBuffer->blocks,curImageBuffer->threads);
			}

			curChunk.x = 0;
		}

		curChunk.y = 0;
	}

	return localChunks;
}


std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, size_t memAvailable, size_t bytesPerVal, size_t maxThreads, Vec<size_t> kernelDims/*=Vec<size_t>(0,0,0)*/)
{
	size_t numVoxels = (size_t)(memAvailable / (bytesPerVal*numBuffersNeeded));
	Vec<size_t> spatialDimensions = imageDims.spatialDimensions;

	Vec<size_t> overlapVolume;
	overlapVolume.x = (kernelDims.x - 1) * spatialDimensions.y * spatialDimensions.z;
	overlapVolume.y = spatialDimensions.x * (kernelDims.y - 1) * spatialDimensions.z;
	overlapVolume.z = spatialDimensions.x * spatialDimensions.y * (kernelDims.z - 1);

	Vec<size_t> deviceSpatialDims(0, 0, 0);

	if (overlapVolume.x > overlapVolume.y && overlapVolume.x > overlapVolume.z) // chunking in X is the worst
	{
		deviceSpatialDims.x = spatialDimensions.x;
		double leftOver = (double)numVoxels / spatialDimensions.x;
		double squareDim = sqrt(leftOver);

		if (overlapVolume.y < overlapVolume.z) // chunking in Y is second worst
		{
			if (squareDim > spatialDimensions.y)
				deviceSpatialDims.y = spatialDimensions.y;
			else
				deviceSpatialDims.y = (size_t)squareDim;

			deviceSpatialDims.z = (size_t)(numVoxels / (deviceSpatialDims.y*deviceSpatialDims.x));

			if (deviceSpatialDims.z > spatialDimensions.z)
			{
				deviceSpatialDims.z = spatialDimensions.z;
				// give some back to y
				deviceSpatialDims.y = (size_t)(numVoxels / (deviceSpatialDims.z*deviceSpatialDims.x));
				deviceSpatialDims.y = MIN(deviceSpatialDims.y, spatialDimensions.y);
			}
		}
		else // chunking in Z is second worst
		{
			if (squareDim > spatialDimensions.z)
				deviceSpatialDims.z = spatialDimensions.z;
			else
				deviceSpatialDims.z = (size_t)squareDim;

			deviceSpatialDims.y = (size_t)(numVoxels / (deviceSpatialDims.z*deviceSpatialDims.x));

			if (deviceSpatialDims.y > spatialDimensions.y)
			{
				deviceSpatialDims.y = spatialDimensions.y;
				// give some back to z
				deviceSpatialDims.z = (size_t)(numVoxels / (deviceSpatialDims.y*deviceSpatialDims.x));
				deviceSpatialDims.z = MIN(deviceSpatialDims.z, spatialDimensions.z);
			}
		}
	}
	else if (overlapVolume.y > overlapVolume.z) // chunking in Y is the worst
	{
		deviceSpatialDims.y = spatialDimensions.y;
		double leftOver = (double)numVoxels / spatialDimensions.y;
		double squareDim = sqrt(leftOver);

		if (overlapVolume.x < overlapVolume.z)
		{
			if (squareDim > spatialDimensions.x)
				deviceSpatialDims.x = spatialDimensions.x;
			else
				deviceSpatialDims.x = (size_t)squareDim;

			deviceSpatialDims.z = (size_t)(numVoxels / (deviceSpatialDims.x*deviceSpatialDims.y));

			if (deviceSpatialDims.z > spatialDimensions.z)
			{
				deviceSpatialDims.z = spatialDimensions.z;
				// give some back to x
				deviceSpatialDims.x = (size_t)(numVoxels / (deviceSpatialDims.z*deviceSpatialDims.x));
				deviceSpatialDims.x = MIN(deviceSpatialDims.x, spatialDimensions.x);
			}
		}
		else
		{
			if (squareDim > spatialDimensions.z)
				deviceSpatialDims.z = spatialDimensions.z;
			else
				deviceSpatialDims.z = (size_t)squareDim;

			deviceSpatialDims.x = (size_t)(numVoxels / (deviceSpatialDims.z*deviceSpatialDims.y));

			if (deviceSpatialDims.x > spatialDimensions.x)
			{
				deviceSpatialDims.x = spatialDimensions.x;
				// give some back to z
				deviceSpatialDims.z = (size_t)(numVoxels / (deviceSpatialDims.y*deviceSpatialDims.x));
				deviceSpatialDims.z = MIN(deviceSpatialDims.z, spatialDimensions.z);
			}
		}
	}
	else // chunking in Z is the worst
	{
		deviceSpatialDims.z = spatialDimensions.z;
		double leftOver = (double)numVoxels / spatialDimensions.z;
		double squareDim = sqrt(leftOver);

		if (overlapVolume.x < overlapVolume.y)
		{
			if (squareDim > spatialDimensions.x)
				deviceSpatialDims.x = spatialDimensions.x;
			else
				deviceSpatialDims.x = (size_t)squareDim;

			deviceSpatialDims.y = (size_t)(numVoxels / (deviceSpatialDims.x*deviceSpatialDims.z));

			if (deviceSpatialDims.y > spatialDimensions.y)
			{
				deviceSpatialDims.y = spatialDimensions.y;
				// give some back to x
				deviceSpatialDims.x = (size_t)(numVoxels / (deviceSpatialDims.z*deviceSpatialDims.x));
				deviceSpatialDims.x = MIN(deviceSpatialDims.x, spatialDimensions.x);
			}
		}
		else
		{
			if (squareDim > spatialDimensions.y)
				deviceSpatialDims.y = spatialDimensions.y;
			else
				deviceSpatialDims.y = (size_t)squareDim;

			deviceSpatialDims.x = (size_t)(numVoxels / (deviceSpatialDims.y*deviceSpatialDims.z));

			if (deviceSpatialDims.x > spatialDimensions.x)
			{
				deviceSpatialDims.x = spatialDimensions.x;
				// give some back to y
				deviceSpatialDims.y = (size_t)(numVoxels / (deviceSpatialDims.z*deviceSpatialDims.x));
				deviceSpatialDims.y = MIN(deviceSpatialDims.y, spatialDimensions.y);
			}
		}
	}

	ImageDimensions deviceDims;
	deviceDims.spatialDimensions = deviceSpatialDims;

	// check to see how many channels will fit on the device
	if (numVoxels>)


	return calculateChunking(imageDims, deviceSpatialDims, maxThreads, kernelDims);
}

