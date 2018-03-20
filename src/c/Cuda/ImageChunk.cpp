#include "ImageChunk.h"
#include "CudaUtilities.h"

void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims)
{
	maxDeviceDims = Vec<size_t>(0, 0, 0);

	for(std::vector<ImageChunk>::iterator curChunk = chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		Vec<size_t> curDim = curChunk->getFullChunkSize();

		if(curDim.x>maxDeviceDims.x)
			maxDeviceDims.x = curDim.x;

		if(curDim.y>maxDeviceDims.y)
			maxDeviceDims.y = curDim.y;

		if(curDim.z>maxDeviceDims.z)
			maxDeviceDims.z = curDim.z;
	}
}

std::vector<ImageChunk> calculateChunking(ImageDimensions imageDims, Vec<size_t> deviceDims, CudaDevices cudaDevs, Vec<size_t> kernalDims /*= Vec<size_t>(1, 1, 1)*/)
{
	std::vector<ImageChunk> localChunks;
	Vec<size_t> margin((kernalDims+1)/2); //integer round
	ImageDimensions chunkDelta(Vec<size_t>(1), 1, 1);
	chunkDelta.dims = MAX(Vec<size_t>(1),deviceDims-margin*2);
	ImageDimensions numChunks;
	numChunks.dims = Vec<size_t>(1, 1, 1);
	numChunks.chan = 1;
	numChunks.frame = 1;

	Vec<size_t> devSpatialDims = deviceDims;

	if(imageDims.dims.x>deviceDims.x)
		numChunks.dims.x = (size_t)ceil((double)imageDims.dims.x/chunkDelta.dims.x);
	else
		chunkDelta.dims.x = imageDims.dims.x;

	if(imageDims.dims.y>deviceDims.y)
		numChunks.dims.y = (size_t)ceil((double)imageDims.dims.y/chunkDelta.dims.y);
	else
		chunkDelta.dims.y = imageDims.dims.y;

	if(imageDims.dims.z>deviceDims.z)
		numChunks.dims.z = (size_t)ceil((double)imageDims.dims.z/chunkDelta.dims.z);
	else
		chunkDelta.dims.z = imageDims.dims.z;

	numChunks.frame = imageDims.frame;
	chunkDelta.frame = 1;
	numChunks.chan = imageDims.chan;
	chunkDelta.chan = 1;

	localChunks.resize(numChunks.getNumElements());

	ImageDimensions curChunk(Vec<size_t>(0), 0, 0);
	ImageDimensions imageStart(Vec<size_t>(0), 0, 0);
	Vec<size_t> chunkROIstart(0, 0, 0);
	Vec<size_t> imageROIstart(0, 0, 0);
	unsigned int chan = 0;
	unsigned int frame = 0;
	ImageDimensions imageEnd(Vec<size_t>(0), 0, 0);
	Vec<size_t> chunkROIend(0, 0, 0);
	Vec<size_t> imageROIend(0, 0, 0);

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

						Vec<size_t> imStartVec = Vec<size_t>(Vec<int>::max(Vec<int>(imageROIstart) - Vec<int>(margin), Vec<int>(0, 0, 0)));
						imageStart = ImageDimensions(imStartVec, curChunk.chan*chunkDelta.chan, curChunk.frame*chunkDelta.frame);
						Vec<size_t> imEndVec = Vec<size_t>::min(imageROIend + margin, imageDims.dims - 1);
						imageEnd = ImageDimensions(imEndVec, imageStart.chan+chunkDelta.chan-1, imageStart.chan + chunkDelta.chan - 1);
						
						chunkROIstart = imageROIstart-imageStart.dims;
						chunkROIend = imageROIend-imageStart.dims;
						
						chan = chunkDelta.chan * curChunk.chan;
						
						frame = chunkDelta.frame* curChunk.frame;

						ImageChunk* curImageBuffer = &localChunks[numChunks.linearAddressAt(curChunk)];

						curImageBuffer->imageROIstart = imageROIstart;
						curImageBuffer->imageROIend = imageROIend;

						curImageBuffer->imageStart = imageStart.dims;
						curImageBuffer->imageEnd = imageEnd.dims;

						curImageBuffer->chunkROIstart = chunkROIstart;
						curImageBuffer->chunkROIend = chunkROIend;

						curImageBuffer->channel = chan;
						curImageBuffer->frame = frame;

						calcBlockThread(curImageBuffer->getFullChunkSize(), cudaDevs.getMaxThreadsPerBlock(), curImageBuffer->blocks, curImageBuffer->threads);
					}
				}
			}
		}
	}

	return localChunks;
}

std::vector<ImageChunk> calculateBuffers(ImageDimensions imageDims, int numBuffersNeeded, CudaDevices cudaDevs, size_t bytesPerVal, Vec<size_t> kernelDims /*= Vec<size_t>(0, 0, 0)*/)
{
	size_t numVoxels = (size_t)(cudaDevs.getMinAvailMem()/(bytesPerVal*numBuffersNeeded));
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

	return calculateChunking(imageDims, deviceDims, cudaDevs, kernelDims);
}


Vec<size_t> ImageChunk::getFullChunkSize()
{
	return imageEnd - imageStart + Vec<size_t>(1);
}

