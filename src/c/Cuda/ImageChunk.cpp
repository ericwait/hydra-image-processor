#include "ImageChunk.h"
#include "CudaUtilities.h"

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, ImageDimensions &maxDeviceDims)
{
	maxDeviceDims.dims = Vec<size_t>(0, 0, 0);

	for(std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		ImageDimensions curDim = curChunk->getFullChunkSize();

		if(curDim.dims.x>maxDeviceDims.dims.x)
			maxDeviceDims.dims.x = curDim.dims.x;

		if(curDim.dims.y>maxDeviceDims.dims.y)
			maxDeviceDims.dims.y = curDim.dims.y;

		if(curDim.dims.z>maxDeviceDims.dims.z)
			maxDeviceDims.dims.z = curDim.dims.z;

		if(curDim.chan>maxDeviceDims.chan)
			maxDeviceDims.chan = curDim.chan;

		if(curDim.frame>maxDeviceDims.frame)
			maxDeviceDims.frame = curDim.frame;
	}
}

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, ImageDimensions deviceDims, size_t maxThreads, Vec<size_t> kernalDims /*= Vec<size_t>(0, 0, 0)*/)
{
	std::vector<ImageChunk> localChunks;
	Vec<size_t> margin((kernalDims+1)/2); //integer round
	ImageDimensions chunkDelta(Vec<size_t>(1), 1, 1);
	chunkDelta.dims = deviceDims.dims-margin*2;
	ImageDimensions numChunks;
	numChunks.dims = Vec<size_t>(1, 1, 1);
	numChunks.chan = 1;
	numChunks.frame = 1;

	Vec<size_t> devSpatialDims = deviceDims.dims;

	if(imageDims.dims.x>deviceDims.dims.x)
		numChunks.dims.x = (size_t)ceil((double)imageDims.dims.x/chunkDelta.x);
	else
		chunkDelta.dims.x = imageDims.dims.x;

	if(imageDims.dims.y>deviceDims.dims.y)
		numChunks.dims.y = (size_t)ceil((double)imageDims.dims.y/chunkDelta.y);
	else
		chunkDelta.dims.y = imageDims.dims.y;

	if(imageDims.dims.z>deviceDims.dims.z)
		numChunks.dims.z = (size_t)ceil((double)imageDims.dims.z/chunkDelta.z);
	else
		chunkDelta.dims.z = imageDims.dims.z;

	if(imageDims.chan>deviceDims.chan)
		numChunks.chan = (size_t)ceil((double)imageDims.chan/deviceDims.chan);
	else
		chunkDelta.chan = imageDims.chan;

	if(imageDims.chan>deviceDims.chan)
		numChunks.chan = (size_t)ceil((double)imageDims.chan/deviceDims.chan);
	else
		chunkDelta.chan = imageDims.chan;

	localChunks.resize(numChunks.getNumElements());

	ImageDimensions curChunk(Vec<size_t>(0), 0, 0);
	Vec<size_t> imageStart(0, 0, 0);
	Vec<size_t> chunkROIstart(0, 0, 0);
	Vec<size_t> imageROIstart(0, 0, 0);
	unsigned int chanStart = 0;
	unsigned int frameStart = 0;
	Vec<size_t> imageEnd(0, 0, 0);
	Vec<size_t> chunkROIend(0, 0, 0);
	Vec<size_t> imageROIend(0, 0, 0);
	unsigned int chanEnd = 0;
	unsigned int frameEnd = 0;

	for(curChunk.frame = 0; curChunk.frame<numChunks.frame; ++curChunk.frame)
	{
		for(curChunk.chan = 0; curChunk.chan<numChunks.chan; ++curChunk.chan)
		{
			for(curChunk.dims.z = 0; curChunk.dims.z<numChunks.dims.z; ++curChunk.dims.z)
			{
				for(curChunk.dims.y = 0; curChunk.dims.y<numChunks.dims.y; ++curChunk.dims.y)
				{
					for(curChunk.dims.x = 0; curChunk.dims.x<numChunks.dims.x; ++curChunk.dims.x)
					{
						imageROIstart = chunkDelta.dims * curChunk.dims;
						imageROIend = Vec<size_t>::min(imageROIstart+chunkDelta.dims, imageDims.dims-1);
						imageStart = Vec<size_t>(Vec<int>::max(Vec<int>(imageROIstart)-Vec<int>(margin), Vec<int>(0, 0, 0)));
						imageEnd = Vec<size_t>::min(imageROIend+margin, imageDims.dims-1);
						chunkROIstart = imageROIstart-imageStart;
						chunkROIend = imageROIend-imageStart;
						chanStart = chunkDelta.chan * curChunk.chan;
						chanEnd = chanStart+chunkDelta.chan-1;
						frameStart = chunkDelta.frame* curChunk.frame;
						frameEnd = frameStart+chunkDelta.frame-1;

						ImageChunk* curImageBuffer = &localChunks[numChunks.linearAddressAt(curChunk)];

						curImageBuffer->imageStart = imageStart;
						curImageBuffer->chunkROIstart = chunkROIstart;
						curImageBuffer->imageROIstart = imageROIstart;
						curImageBuffer->imageEnd = imageEnd;
						curImageBuffer->chunkROIend = chunkROIend;
						curImageBuffer->imageROIend = imageROIend;
						curImageBuffer->channelStart = chanStart;
						curImageBuffer->channelEnd = chanEnd;
						curImageBuffer->frameStart = frameStart;
						curImageBuffer->frameEnd = frameEnd;

						calcBlockThread(curImageBuffer->getFullChunkSize().dims, maxThreads, curImageBuffer->blocks, curImageBuffer->threads);
					}
				}
			}
		}
	}

	return localChunks;
}

std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, size_t memAvailable, size_t bytesPerVal, size_t maxThreads, Vec<size_t> kernelDims /*= Vec<size_t>(0, 0, 0)*/)
{
	size_t numVoxels = (size_t)(memAvailable/(bytesPerVal*numBuffersNeeded));
	Vec<size_t> dims = imageDims.dims;

	Vec<size_t> overlapVolume;
	overlapVolume.x = (kernelDims.x-1) * dims.y * dims.z;
	overlapVolume.y = dims.x * (kernelDims.y-1) * dims.z;
	overlapVolume.z = dims.x * dims.y * (kernelDims.z-1);

	Vec<size_t> deviceDims(0, 0, 0);

	if(overlapVolume.x>overlapVolume.y && overlapVolume.x>overlapVolume.z) // chunking in X is the worst
	{
		deviceDims.x = dims.x;
		double leftOver = (double)numVoxels/dims.x;
		double squareDim = sqrt(leftOver);

		if(overlapVolume.y<overlapVolume.z) // chunking in Y is second worst
		{
			if(squareDim>dims.y)
				deviceDims.y = dims.y;
			else
				deviceDims.y = (size_t)squareDim;

			deviceDims.z = (size_t)(numVoxels/(deviceDims.y*deviceDims.x));

			if(deviceDims.z>dims.z)
			{
				deviceDims.z = dims.z;
				// give some back to y
				deviceDims.y = (size_t)(numVoxels/(deviceDims.z*deviceDims.x));
				deviceDims.y = MIN(deviceDims.y, dims.y);
			}
		} else // chunking in Z is second worst
		{
			if(squareDim>dims.z)
				deviceDims.z = dims.z;
			else
				deviceDims.z = (size_t)squareDim;

			deviceDims.y = (size_t)(numVoxels/(deviceDims.z*deviceDims.x));

			if(deviceDims.y>dims.y)
			{
				deviceDims.y = dims.y;
				// give some back to z
				deviceDims.z = (size_t)(numVoxels/(deviceDims.y*deviceDims.x));
				deviceDims.z = MIN(deviceDims.z, dims.z);
			}
		}
	} else if(overlapVolume.y>overlapVolume.z) // chunking in Y is the worst
	{
		deviceDims.y = dims.y;
		double leftOver = (double)numVoxels/dims.y;
		double squareDim = sqrt(leftOver);

		if(overlapVolume.x<overlapVolume.z)
		{
			if(squareDim>dims.x)
				deviceDims.x = dims.x;
			else
				deviceDims.x = (size_t)squareDim;

			deviceDims.z = (size_t)(numVoxels/(deviceDims.x*deviceDims.y));

			if(deviceDims.z>dims.z)
			{
				deviceDims.z = dims.z;
				// give some back to x
				deviceDims.x = (size_t)(numVoxels/(deviceDims.z*deviceDims.x));
				deviceDims.x = MIN(deviceDims.x, dims.x);
			}
		} else
		{
			if(squareDim>dims.z)
				deviceDims.z = dims.z;
			else
				deviceDims.z = (size_t)squareDim;

			deviceDims.x = (size_t)(numVoxels/(deviceDims.z*deviceDims.y));

			if(deviceDims.x>dims.x)
			{
				deviceDims.x = dims.x;
				// give some back to z
				deviceDims.z = (size_t)(numVoxels/(deviceDims.y*deviceDims.x));
				deviceDims.z = MIN(deviceDims.z, dims.z);
			}
		}
	} else // chunking in Z is the worst
	{
		deviceDims.z = dims.z;
		double leftOver = (double)numVoxels/dims.z;
		double squareDim = sqrt(leftOver);

		if(overlapVolume.x<overlapVolume.y)
		{
			if(squareDim>dims.x)
				deviceDims.x = dims.x;
			else
				deviceDims.x = (size_t)squareDim;

			deviceDims.y = (size_t)(numVoxels/(deviceDims.x*deviceDims.z));

			if(deviceDims.y>dims.y)
			{
				deviceDims.y = dims.y;
				// give some back to x
				deviceDims.x = (size_t)(numVoxels/(deviceDims.z*deviceDims.x));
				deviceDims.x = MIN(deviceDims.x, dims.x);
			}
		} else
		{
			if(squareDim>dims.y)
				deviceDims.y = dims.y;
			else
				deviceDims.y = (size_t)squareDim;

			deviceDims.x = (size_t)(numVoxels/(deviceDims.y*deviceDims.z));

			if(deviceDims.x>dims.x)
			{
				deviceDims.x = dims.x;
				// give some back to y
				deviceDims.y = (size_t)(numVoxels/(deviceDims.z*deviceDims.x));
				deviceDims.y = MIN(deviceDims.y, dims.y);
			}
		}
	}

	ImageDimensions deviceImageDims;
	deviceImageDims.dims = deviceDims;

	// check to see how many channels will fit on the device
	size_t numVoxelsPerDevice = deviceDims.product();
	deviceImageDims.chan = MAX(1, numVoxels/numVoxelsPerDevice);
	deviceImageDims.frame = MAX(1, numVoxels/(deviceImageDims.chan*numVoxelsPerDevice));

	return calculateChunking(imageDims, deviceImageDims, maxThreads, kernelDims);
}
