#include "CudaUtilities.cuh"
#include "CudaKernels.cuh"
#include "CudaProcessBuffer.cuh"
#include "CudaDeviceImages.cuh"
#include "CHelpers.h"

//Percent of memory that can be used on the device
const double MAX_MEM_AVAIL = 0.95;

std::vector<ImageChunk> calculateBuffers(Vec<size_t> imageDims, int numBuffersNeeded, size_t memAvailable, const cudaDeviceProp& prop,
										 Vec<size_t> kernalDims/*=Vec<size_t>(0,0,0)*/)
{
	size_t numVoxels = (size_t)(memAvailable / (sizeof(HostPixelType)*numBuffersNeeded));

	Vec<size_t> overlapVolume;
	overlapVolume.x = kernalDims.x * imageDims.y * imageDims.z;
	overlapVolume.y = imageDims.x * kernalDims.y * imageDims.z;
	overlapVolume.z = imageDims.x * imageDims.y * kernalDims.z;

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

	return calculateChunking(imageDims, deviceDims, prop, kernalDims);
}

std::vector<ImageChunk> calculateChunking(Vec<size_t> orgImageDims, Vec<size_t> deviceDims, const cudaDeviceProp& prop,
										  Vec<size_t> kernalDims/*=Vec<size_t>(0,0,0)*/)
{
	std::vector<ImageChunk> localChunks;
	Vec<size_t> margin((kernalDims + 1)/2); //integer round
	Vec<size_t> chunkDelta(deviceDims-margin*2);
	Vec<size_t> numChunks(1,1,1);

	if (orgImageDims.x>deviceDims.x)
		numChunks.x = (size_t)ceil((double)orgImageDims.x/chunkDelta.x);
	else
		chunkDelta.x = orgImageDims.x;

	if (orgImageDims.y>deviceDims.y)
		numChunks.y = (size_t)ceil((double)orgImageDims.y/chunkDelta.y);
	else
		chunkDelta.y = orgImageDims.y;

	if (orgImageDims.z>deviceDims.z)
		numChunks.z = (size_t)ceil((double)orgImageDims.z/chunkDelta.z);
	else
		chunkDelta.z = orgImageDims.z;

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
				imageROIend = Vec<size_t>::min(imageROIstart + chunkDelta, orgImageDims);
				imageStart = Vec<size_t>(Vec<int>::max(Vec<int>(imageROIstart)-Vec<int>(margin), Vec<int>(0,0,0)));
				imageEnd = Vec<size_t>::min(imageROIend + margin, orgImageDims);
				chunkROIstart = imageROIstart - imageStart;
				chunkROIend = imageROIend - imageStart;

				ImageChunk* curImageBuffer = &localChunks[numChunks.linearAddressAt(curChunk)];

				curImageBuffer->imageStart = imageStart;
				curImageBuffer->chunkROIstart = chunkROIstart;
				curImageBuffer->imageROIstart = imageROIstart;
				curImageBuffer->imageEnd = imageEnd;
				curImageBuffer->chunkROIend = chunkROIend;
				curImageBuffer->imageROIend = imageROIend;

				calcBlockThread(curImageBuffer->getFullChunkSize(),prop,curImageBuffer->blocks,curImageBuffer->threads);
			}

			curChunk.x = 0;
		}

		curChunk.y = 0;
	}

	return localChunks;
}

CudaProcessBuffer::CudaProcessBuffer(int device/*=0*/)
{
	defaults();
	this->device = device;
	deviceSetup();
}

CudaProcessBuffer::~CudaProcessBuffer()
{
	defaults();
}

void CudaProcessBuffer::deviceSetup()
{
	HANDLE_ERROR(cudaSetDevice(device));
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
}

void CudaProcessBuffer::defaults()
{
	device = 0;
	orgImageDims = Vec<size_t>(0,0,0);
	maxDeviceDims = Vec<size_t>(0,0,0);
}

//////////////////////////////////////////////////////////////////////////
// Helper Functions
//////////////////////////////////////////////////////////////////////////

void CudaProcessBuffer::setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims)
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

void runGaussIterations(Vec<int> &gaussIterations, std::vector<ImageChunk>::iterator& curChunk, CudaDeviceImages& deviceImages,
						Vec<size_t> sizeconstKernelDims)
{
	for (int x=0; x<gaussIterations.x; ++x)
	{
		cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			Vec<size_t>(sizeconstKernelDims.x,1,1));
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();
	}

	for (int y=0; y<gaussIterations.y; ++y)
	{
		cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			Vec<size_t>(1,sizeconstKernelDims.y,1),	sizeconstKernelDims.x);
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();
	}

	for (int z=0; z<gaussIterations.z; ++z)
	{
		cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			Vec<size_t>(1,1,sizeconstKernelDims.z),	sizeconstKernelDims.y);
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();
	}
}

void runMedianFilter(cudaDeviceProp& deviceProp, std::vector<ImageChunk>::iterator curChunk, Vec<size_t> &neighborhood, 
					 CudaDeviceImages& deviceImages)
{
	dim3 blocks(curChunk->blocks);
	dim3 threads(curChunk->threads);
	double threadVolume = threads.x * threads.y * threads.z;
	double newThreadVolume = (double)deviceProp.sharedMemPerBlock/(sizeof(DevicePixelType)*neighborhood.product());

	if (newThreadVolume<threadVolume)
	{
		double alpha = pow(threadVolume/newThreadVolume,1.0/3.0);
		threads.x = (unsigned int)(threads.x / alpha);
		threads.y = (unsigned int)(threads.y / alpha);
		threads.z = (unsigned int)(threads.z / alpha);
		threads.x = (threads.x>0) ? (threads.x) : (1);
		threads.y = (threads.y>0) ? (threads.y) : (1);
		threads.z = (threads.z>0) ? (threads.z) : (1);

		blocks.x = (unsigned int)ceil((double)curChunk->getFullChunkSize().x / threads.x);
		blocks.y = (unsigned int)ceil((double)curChunk->getFullChunkSize().y / threads.y);
		blocks.z = (unsigned int)ceil((double)curChunk->getFullChunkSize().z / threads.z);
	}

	size_t sharedMemorysize = neighborhood.product()*sizeof(DevicePixelType) * threads.x * threads.y * threads.z;

	cudaMedianFilter<<<blocks,threads,sharedMemorysize>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),neighborhood);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();
}

DevicePixelType* CudaProcessBuffer::setUpOutIm(Vec<size_t> dims, DevicePixelType** imageOut)
{
	orgImageDims = dims;

	DevicePixelType* imOut;
	if (imageOut==NULL)
		imOut = new DevicePixelType[orgImageDims.product()];
	else
		imOut = *imageOut;

	return imOut;
}

//////////////////////////////////////////////////////////////////////////
//Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

DevicePixelType* CudaProcessBuffer::addConstant(const DevicePixelType* imageIn, Vec<size_t> dims, double additive,
												DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaAddFactor<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			additive,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::addImageWith(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims,
													  double additive, DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	std::vector<ImageChunk> chunks = calculateBuffers(dims,3,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(3,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());
		curChunk->sendROI(imageIn1,dims,deviceImages.getCurBuffer());
		curChunk->sendROI(imageIn2,dims,deviceImages.getNextBuffer());

		cudaAddTwoImagesWithFactor<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			*(deviceImages.getThirdBuffer()),additive,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		curChunk->retriveROI(imOut,dims,deviceImages.getThirdBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::applyPolyTransformation(const DevicePixelType* imageIn, Vec<size_t> dims, double a, double b, double c,
												DevicePixelType minValue, DevicePixelType maxValue, DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);
	
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaPolyTransferFuncImage<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			a,b,c,minValue,maxValue);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::contrastEnhancement(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas,
														Vec<size_t> neighborhood, DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);

	Vec<int> gaussIterations(0,0,0);
	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,hostKernel,gaussIterations);
	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));

	std::vector<ImageChunk> chunks = calculateBuffers(dims,3,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,
		sizeconstKernelDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(3,maxDeviceDims,device);
 
	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());

		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());

		runGaussIterations(gaussIterations, curChunk, deviceImages, sizeconstKernelDims);

		curChunk->sendROI(imageIn,dims,deviceImages.getNextBuffer());

		cudaAddTwoImagesWithFactor<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			*(deviceImages.getThirdBuffer()),-1.0,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.setNthBuffCurent(3);

		runMedianFilter(deviceProp, curChunk, neighborhood, deviceImages);

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

size_t* CudaProcessBuffer::createHistogram(const DevicePixelType* imageIn, Vec<size_t> dims, int& arraySize)
{
	arraySize = NUM_BINS;
	size_t* hostHist = new size_t[arraySize];

	size_t* deviceHist;
	HANDLE_ERROR(cudaMalloc((void**)&deviceHist,sizeof(size_t)*arraySize));
	HANDLE_ERROR(cudaMemset(deviceHist,0,sizeof(size_t)*arraySize));

	std::vector<ImageChunk> chunks = calculateBuffers(dims,1,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);
	setMaxDeviceDims(chunks, maxDeviceDims);
	CudaDeviceImages deviceImages(1,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		
		cudaHistogramCreate<<<deviceProp.multiProcessorCount*2,arraySize,sizeof(size_t)*arraySize>>>(*(deviceImages.getCurBuffer()),
			deviceHist);
		DEBUG_KERNEL_CHECK();
	}
	HANDLE_ERROR(cudaMemcpy(hostHist,deviceHist,sizeof(size_t)*arraySize,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceHist));

	return hostHist;
}

DevicePixelType* CudaProcessBuffer::gaussianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas,
												   DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	Vec<int> gaussIterations(0,0,0);
	sigmas.x = (dims.x==1) ? (0) : (sigmas.x);
	sigmas.y = (dims.y==1) ? (0) : (sigmas.y);
	sigmas.z = (dims.z==1) ? (0) : (sigmas.z);

	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,hostKernel,gaussIterations);
	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,
		sizeconstKernelDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());

		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());

		runGaussIterations(gaussIterations, curChunk, deviceImages, sizeconstKernelDims);

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::maxFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernalDims, float* kernel/*=NULL*/,
						   DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	if (kernel==NULL)
	{
		kernalDims = kernalDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernalDims.product()];
		memset(ones,1,kernalDims.product());
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(float)*kernalDims.product()));
		delete[] ones;
	} 
	else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*kernalDims.product()));
	}

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,kernalDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMaxFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernalDims,
			minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::meanFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood,
											 DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,neighborhood);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());
		
		cudaMeanFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),neighborhood);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();
		
		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}
	
	return imOut;
}

DevicePixelType* CudaProcessBuffer::medianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood,
												 DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,neighborhood);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		runMedianFilter(deviceProp, curChunk, neighborhood, deviceImages);

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::minFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernalDims, float* kernel/*=NULL*/,
											  DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	if (kernel==NULL)
	{
		kernalDims = kernalDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernalDims.product()];
		memset(ones,1,kernalDims.product());
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(float)*kernalDims.product()));
		delete[] ones;
	} 
	else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*kernalDims.product()));
	}

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,kernalDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMinFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernalDims,
			minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::multiplyImage(const DevicePixelType* imageIn, Vec<size_t> dims, double multiplier, 
												  DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMultiplyImage<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			multiplier,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

DevicePixelType* CudaProcessBuffer::multiplyImageWith(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims,
													  double factor, DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	std::vector<ImageChunk> chunks = calculateBuffers(dims,3,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(3,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());
		curChunk->sendROI(imageIn1,dims,deviceImages.getCurBuffer());
		curChunk->sendROI(imageIn2,dims,deviceImages.getNextBuffer());

		cudaMultiplyTwoImages<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			*(deviceImages.getThirdBuffer()),factor,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		curChunk->retriveROI(imOut,dims,deviceImages.getThirdBuffer());
	}

	return imOut;
}

double CudaProcessBuffer::normalizedCovariance(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims)
{
// 	double im1Mean = sumArray(imageIn1,dims.product()) / dims.product();
// 	double im2Mean = sumArray(imageIn2,dims.product()) / dims.product();
// 
// 	DevicePixelType* im1Sub = addConstant(imageIn1,dims,-1.0*im1Mean);
// 	DevicePixelType* im2Sub = addConstant(imageIn2,dims,-1.0*im2Mean);
// 
// 	DevicePixelType* im1P = imagePow(im1Sub,dims,2.0);
// 	DevicePixelType* im2P = imagePow(im2Sub,dims,2.0);
// 
// 	double sigma1 = sqrt(sumArray(im1P,dims.product())/dims.product());
// 	double sigma2 = sqrt(sumArray(im2P,dims.product())/dims.product());
// 
// 	DevicePixelType* imMul = multiplyImageWith(im1Sub,im2Sub,dims,1.0);
// 	double numarator = sumArray(imMul,dims.product());
// 
// 	double coVar = numarator/(dims.product()*sigma1*sigma2);
// 
// 	delete[] im1Sub;
// 	delete[] im2Sub;
// 	delete[] im1P;
// 	delete[] im2P;
// 	delete[] imMul;
// 
// 	return coVar;

	return 0.0;
}

double* CudaProcessBuffer::normalizeHistogram(const DevicePixelType* imageIn, Vec<size_t> dims, int& arraySize)
{
	arraySize = NUM_BINS;
	double* hostHistNorm = new double[arraySize];

	size_t* deviceHist;
	double* deviceHistNorm;
	
	checkFreeMemory(sizeof(size_t)*arraySize+sizeof(double)*arraySize,device,true);

	HANDLE_ERROR(cudaMalloc((void**)&deviceHist,sizeof(size_t)*arraySize));
	HANDLE_ERROR(cudaMalloc((void**)&deviceHistNorm,sizeof(double)*arraySize));
	HANDLE_ERROR(cudaMemset(deviceHist,0,sizeof(size_t)*arraySize));

	std::vector<ImageChunk> chunks = calculateBuffers(dims,1,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);
	setMaxDeviceDims(chunks, maxDeviceDims);
	CudaDeviceImages deviceImages(1,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());

		cudaHistogramCreate<<<deviceProp.multiProcessorCount*2,arraySize,sizeof(size_t)*arraySize>>>(*(deviceImages.getCurBuffer()),
			deviceHist);
		DEBUG_KERNEL_CHECK();
	}

	cudaNormalizeHistogram<<<arraySize,1>>>(deviceHist,deviceHistNorm,dims);
	DEBUG_KERNEL_CHECK();

	HANDLE_ERROR(cudaMemcpy(hostHistNorm,deviceHistNorm,sizeof(double)*arraySize,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceHist));

	return hostHistNorm;
}

DevicePixelType* CudaProcessBuffer::otsuThresholdFilter(const DevicePixelType* imageIn, Vec<size_t> dims, double alpha/*=1.0*/,
														DevicePixelType** imageOut/*=NULL*/)
{
	double thresh = otsuThresholdValue(imageIn,dims);
	thresh *= alpha;

	return thresholdFilter(imageIn,dims,(DevicePixelType)thresh,imageOut);
}

double CudaProcessBuffer::otsuThresholdValue(const DevicePixelType* imageIn, Vec<size_t> dims)
{
	int arraySize;
	double* hist = normalizeHistogram(imageIn,dims,arraySize);

	double thrsh = calcOtsuThreshold(hist,arraySize);

	delete[] hist;

	return thrsh;
}

DevicePixelType* CudaProcessBuffer::imagePow(const DevicePixelType* imageIn, Vec<size_t> dims, double power, DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaPow<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			power,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

double CudaProcessBuffer::sumArray(const DevicePixelType* imageIn, size_t n)
{
	double sum = 0.0;
	double* deviceSum;
	double* hostSum;
	DevicePixelType* deviceImage;

	unsigned int blocks = deviceProp.multiProcessorCount;
	unsigned int threads = deviceProp.maxThreadsPerBlock;

	Vec<size_t> maxDeviceDims(1,1,1);

	maxDeviceDims.x = (n < (double)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL)/sizeof(DevicePixelType)) ? (n) :
		((size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL/sizeof(DevicePixelType)));

	checkFreeMemory(sizeof(DevicePixelType)*maxDeviceDims.x+sizeof(double)*blocks,device,true);
	HANDLE_ERROR(cudaMalloc((void**)&deviceImage,sizeof(DevicePixelType)*maxDeviceDims.x));
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*blocks));
	hostSum = new double[blocks];

	for (int i=0; i<ceil((double)n/maxDeviceDims.x); ++i)
	{
		const DevicePixelType* imStart = imageIn + i*maxDeviceDims.x;
		size_t numValues = ((i+1)*maxDeviceDims.x < n) ? (maxDeviceDims.x) : (n-i*maxDeviceDims.x);

		HANDLE_ERROR(cudaMemcpy(deviceImage,imStart,sizeof(DevicePixelType)*numValues,cudaMemcpyHostToDevice));

		cudaSumArray<<<blocks,threads,sizeof(double)*threads>>>(deviceImage,deviceSum,numValues);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*blocks,cudaMemcpyDeviceToHost));

		for (unsigned int i=0; i<blocks; ++i)
		{
			sum += hostSum[i];
		}
	}

	HANDLE_ERROR(cudaFree(deviceSum));
	HANDLE_ERROR(cudaFree(deviceImage));

	delete[] hostSum;

	return sum;
}

DevicePixelType* CudaProcessBuffer::reduceImage(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> reductions,
												Vec<size_t>& reducedDims, DevicePixelType** imageOut/*=NULL*/)
{
	reductions = reductions.clamp(Vec<size_t>(1,1,1),dims);
	orgImageDims = dims;
	reducedDims = orgImageDims / reductions;
	DevicePixelType* reducedImage;
	if (imageOut==NULL)
		reducedImage = new DevicePixelType[reducedDims.product()];
	else
		reducedImage = *imageOut;

	double ratio = (double)reducedDims.product() / dims.product();

	if (ratio==1.0)
	{
		memcpy(reducedImage,imageIn,sizeof(DevicePixelType)*reducedDims.product());
		return reducedImage;
	}

	std::vector<ImageChunk> orgChunks = calculateBuffers(dims,1,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL*(1-ratio)),deviceProp,reductions);
	std::vector<ImageChunk> reducedChunks = orgChunks;

	for (std::vector<ImageChunk>::iterator it=reducedChunks.begin(); it!=reducedChunks.end(); ++it)
	{
		it->imageStart = it->imageROIstart/reductions;
		it->chunkROIstart = Vec<size_t>(0,0,0);
		it->imageROIstart = it->imageROIstart/reductions;
		it->imageEnd = it->imageROIend/reductions;
		it->imageROIend = it->imageROIend/reductions;
		it->chunkROIend = it->imageEnd-it->imageStart;

		calcBlockThread(it->getFullChunkSize(),deviceProp,it->blocks,it->threads);
	}

	CudaImageContainerClean* deviceImageIn = new CudaImageContainerClean(orgChunks[0].getFullChunkSize(),device);
	CudaImageContainerClean* deviceImageOut = new CudaImageContainerClean(reducedChunks[0].getFullChunkSize(),device);

	std::vector<ImageChunk>::iterator orgIt = orgChunks.begin();
	std::vector<ImageChunk>::iterator reducedIt = reducedChunks.begin();

	while (orgIt!=orgChunks.end() && reducedIt!=reducedChunks.end())
	{
		orgIt->sendROI(imageIn,dims,deviceImageIn);
		deviceImageOut->setDims(reducedIt->getFullChunkSize());

		dim3 blocks(reducedIt->blocks);
		dim3 threads(reducedIt->threads);
 		double threadVolume = threads.x * threads.y * threads.z;
 		double newThreadVolume = (double)deviceProp.sharedMemPerBlock/(sizeof(DevicePixelType)*reductions.product());
 
		if (newThreadVolume<threadVolume)
		{
			double alpha = pow(threadVolume/newThreadVolume,1.0/3.0);
			threads.x = (unsigned int)(threads.x / alpha);
			threads.y = (unsigned int)(threads.y / alpha);
			threads.z = (unsigned int)(threads.z / alpha);
			threads.x = (threads.x>0) ? (threads.x) : (1);
			threads.y = (threads.y>0) ? (threads.y) : (1);
			threads.z = (threads.z>0) ? (threads.z) : (1);

			if (threads.x*threads.y*threads.z>(unsigned int)deviceProp.maxThreadsPerBlock)
			{
				unsigned int maxThreads = (unsigned int)pow(deviceProp.maxThreadsPerBlock,1.0/3.0);
				threads.x = maxThreads;
				threads.y = maxThreads;
				threads.z = maxThreads;
			}


			blocks.x = (unsigned int)ceil((double)reducedIt->getFullChunkSize().x / threads.x);
			blocks.y = (unsigned int)ceil((double)reducedIt->getFullChunkSize().y / threads.y);
			blocks.z = (unsigned int)ceil((double)reducedIt->getFullChunkSize().z / threads.z);
		}
 
 		size_t sharedMemorysize = reductions.product()*sizeof(DevicePixelType) * threads.x * threads.y * threads.z;
 
 		cudaMedianImageReduction<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions);
		DEBUG_KERNEL_CHECK();

		reducedIt->retriveROI(reducedImage,reducedDims,deviceImageOut);
		
		++orgIt;
		++reducedIt;
	}

	delete deviceImageIn;
	delete deviceImageOut;

	cudaThreadExit();

 	return reducedImage;
}

DevicePixelType* CudaProcessBuffer::thresholdFilter(const DevicePixelType* imageIn, Vec<size_t> dims, DevicePixelType thresh,
													DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaThresholdImage<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
			thresh,minVal,maxVal);
		DEBUG_KERNEL_CHECK();

		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	return imOut;
}

void CudaProcessBuffer::unmix(const DevicePixelType* image, Vec<size_t> neighborhood)
{
	//neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);
	throw std::logic_error("The method or operation is not implemented.");
}
