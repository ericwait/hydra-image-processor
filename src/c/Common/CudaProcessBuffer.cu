#include "CudaUtilities.cuh"
#include "CudaKernels.cuh"
#include "CudaProcessBuffer.cuh"

//Percent of memory that can be used on the device
const double MAX_MEM_AVAIL = 0.95;

std::vector<ImageChunk> calculateBuffers(Vec<size_t> imageDims, int numBuffersNeeded, size_t memAvailable, const cudaDeviceProp& prop, Vec<size_t> kernalDims/*=Vec<size_t>(0,0,0)*/)
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

std::vector<ImageChunk> calculateChunking(Vec<size_t> orgImageDims, Vec<size_t> deviceDims, const cudaDeviceProp& prop, Vec<size_t> kernalDims/*=Vec<size_t>(0,0,0)*/)
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
}

//////////////////////////////////////////////////////////////////////////
//Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

DevicePixelType* CudaProcessBuffer::addConstant(const DevicePixelType* imageIn, Vec<size_t> dims, double additive,
												DevicePixelType** imageOut/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
// 	orgImageDims = dims;
// 
// 	DevicePixelType* imOut;
// 	if (imageOut==NULL)
// 		imOut = new DevicePixelType[orgImageDims.product()];
// 	else
// 		imOut = *imageOut;
// 
// 	calculateBufferDims(TODO,2);
// 
// 	while (loadChunk(imageIn, TODO))
// 	{
// 		if (!getNextBuffer()->setDims(getCurrentBuffer()->getDims()))
// 			throw std::runtime_error("Unable to load chunk to the device because the buffer was too small!");
// 
// 		cudaAddFactor<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),additive,std::numeric_limits<DevicePixelType>::min(),std::numeric_limits<DevicePixelType>::max());
// 		incrementBufferNumber();
// 		retriveCurChunk();
// 	}
// 
// 	saveChunks(imOut);
// 
// 	return imOut;
}

DevicePixelType* CudaProcessBuffer::addImageWith(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims,
													  double additive, DevicePixelType** imageOut/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::applyPolyTransformation(double a, double b, double c, DevicePixelType minValue, DevicePixelType maxValue)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::calculateMinMax(double& minValue, double& maxValue)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::contrastEnhancement(Vec<float> sigmas, Vec<size_t> medianNeighborhood)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::createHistogram()
{
	throw std::logic_error("The method or operation is not implemented.");
}

DevicePixelType* CudaProcessBuffer::gaussianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas,DevicePixelType** imageOut/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
// 	DevicePixelType* gaussImage;
// 
// 	createDeviceBuffers(dims, 2);
// 
// 	if (dims==deviceDims)
// 	{
// 		deviceImageBuffers[0]->loadImage(imageIn,dims);
// 		currentBufferIdx = 0;
// 	}
// 	else
// 		throw std::logic_error("Image size not handled yet.");
// 
// 	if (imageOut==NULL)
// 		gaussImage = new DevicePixelType[deviceDims.product()];
// 	else
// 		gaussImage = *imageOut;
// 
// 	Vec<int> gaussIterations(0,0,0);
// 	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,hostKernel,gaussIterations);
// 	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
// 		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));
// 
// 	for (int x=0; x<gaussIterations.x; ++x)
// 	{
// 		cudaMultAddFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),Vec<size_t>(sizeconstKernelDims.x,1,1));
// 		incrementBufferNumber();
// #ifdef _DEBUG
// 		cudaThreadSynchronize();
// 		gpuErrchk( cudaPeekAtLastError() );
// #endif // _DEBUG
// 	}
// 
// 	for (int y=0; y<gaussIterations.y; ++y)
// 	{
// 		cudaMultAddFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),Vec<size_t>(1,sizeconstKernelDims.y,1),
// 			sizeconstKernelDims.x);
// 		incrementBufferNumber();
// #ifdef _DEBUG
// 		cudaThreadSynchronize();
// 		gpuErrchk( cudaPeekAtLastError() );
// #endif // _DEBUG
// 	}
// 
// 	for (int z=0; z<gaussIterations.z; ++z)
// 	{
// 		cudaMultAddFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),Vec<size_t>(1,1,sizeconstKernelDims.z),
// 			sizeconstKernelDims.x+sizeconstKernelDims.y);
// 		incrementBufferNumber();
// #ifdef _DEBUG
// 		cudaThreadSynchronize();
// 		gpuErrchk( cudaPeekAtLastError() );
// #endif // _DEBUG
// 	}
// 
// 	HANDLE_ERROR(cudaMemcpy(gaussImage,getCurrentBuffer()->getDeviceImagePointer(),sizeof(DevicePixelType)*dims.product(),
// 		cudaMemcpyDeviceToHost));
// 
// 	return gaussImage;
	//return NULL;
}

void CudaProcessBuffer::mask(const DevicePixelType* imageMask, DevicePixelType threshold/*=1*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::maxFilter(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::maximumIntensityProjection()
{
	throw std::logic_error("The method or operation is not implemented.");
}

DevicePixelType* CudaProcessBuffer::meanFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood,
											 DevicePixelType** imageOut/*=NULL*/)
{
	orgImageDims = dims;

	DevicePixelType* meanImage;
	if (imageOut==NULL)
		meanImage = new DevicePixelType[orgImageDims.product()];
	else
		meanImage = *imageOut;

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,neighborhood);

	CudaImageContainerClean* deviceImageIn = new CudaImageContainerClean(chunks[0].getFullChunkSize(),device);
	CudaImageContainerClean* deviceImageOut = new CudaImageContainerClean(chunks[0].getFullChunkSize(),device);
	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImageIn);
		deviceImageOut->setDims(curChunk->getFullChunkSize());
		
		cudaMeanFilter<<<curChunk->blocks,curChunk->threads>>>(*deviceImageIn,*deviceImageOut,neighborhood);
		
		curChunk->retriveROI(meanImage,dims,deviceImageOut);
	}
	
	delete deviceImageIn;
	delete deviceImageOut;

	return meanImage;
}

DevicePixelType* CudaProcessBuffer::medianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, DevicePixelType** imageOut/*=NULL*/)
{
	orgImageDims = dims;

	DevicePixelType* medianImage;
	if (imageOut==NULL)
		medianImage = new DevicePixelType[orgImageDims.product()];
	else
		medianImage = *imageOut;

	std::vector<ImageChunk> chunks = calculateBuffers(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,neighborhood);

	CudaImageContainerClean* deviceImageIn = new CudaImageContainerClean(chunks[0].getFullChunkSize(),device);
	CudaImageContainerClean* deviceImageOut = new CudaImageContainerClean(chunks[0].getFullChunkSize(),device);
	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImageIn);
		deviceImageOut->setDims(curChunk->getFullChunkSize());

		dim3 blocks(curChunk->blocks);
		dim3 threads(curChunk->threads);
		double threadVolume = threads.x * threads.y * threads.z;
		double newThreadVolume = (double)deviceProp.sharedMemPerBlock/(sizeof(DevicePixelType)*neighborhood.product());

		double alpha = pow(threadVolume/newThreadVolume,1.0/3.0);
		threads.x = threads.x / alpha;
		threads.y = threads.y / alpha;
		threads.z = threads.z / alpha;

		blocks.x = ceil((double)curChunk->getFullChunkSize().x / threads.x);
		blocks.y = ceil((double)curChunk->getFullChunkSize().y / threads.y);
		blocks.z = ceil((double)curChunk->getFullChunkSize().z / threads.z);

		size_t sharedMemorysize = neighborhood.product() * threads.x * threads.y * threads.z;

 		cudaMedianFilter<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn,*deviceImageOut,neighborhood);
 
 		curChunk->retriveROI(medianImage,dims,deviceImageOut);
	}

	delete deviceImageIn;
	delete deviceImageOut;

	return medianImage;
}

void CudaProcessBuffer::minFilter(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::morphClosure(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::morphOpening(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::multiplyImage(double factor)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::multiplyImageWith(const DevicePixelType* image)
{
	throw std::logic_error("The method or operation is not implemented.");
}

double CudaProcessBuffer::normalizedCovariance(DevicePixelType* otherImage)
{
	throw std::logic_error("The method or operation is not implemented.");
	//return 0.0;
}

void CudaProcessBuffer::normalizeHistogram()
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::otsuThresholdFilter(float alpha/*=1.0f*/)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::imagePow(int p)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::sumArray(double& sum)
{
	throw std::logic_error("The method or operation is not implemented.");
}

DevicePixelType* CudaProcessBuffer::reduceImage(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> reductions,
												Vec<size_t>& reducedDims, DevicePixelType** imageOut/*=NULL*/)
{
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
	size_t sharedMemorysize = 0;
	while (orgIt!=orgChunks.end() && reducedIt!=reducedChunks.end())
	{
		orgIt->sendROI(imageIn,dims,deviceImageIn);
		deviceImageOut->setDims(reducedIt->getFullChunkSize());

		dim3 blocks(reducedIt->blocks);
		dim3 threads(reducedIt->threads);
 		double threadVolume = threads.x * threads.y * threads.z;
 		double newThreadVolume = (double)deviceProp.sharedMemPerBlock/(sizeof(DevicePixelType)*reductions.product());
 
 		double alpha = pow(threadVolume/newThreadVolume,1.0/3.0);
 		threads.x = threads.x / alpha;
 		threads.y = threads.y / alpha;
 		threads.z = threads.z / alpha;

		if (threads.x*threads.y*threads.z>deviceProp.maxThreadsPerBlock)
		{
			unsigned int maxThreads = pow(deviceProp.maxThreadsPerBlock,1.0/3.0);
			threads.x = maxThreads;
			threads.y = maxThreads;
			threads.z = maxThreads;
		}
 
 		blocks.x = ceil((double)reducedIt->getFullChunkSize().x / threads.x);
 		blocks.y = ceil((double)reducedIt->getFullChunkSize().y / threads.y);
 		blocks.z = ceil((double)reducedIt->getFullChunkSize().z / threads.z);
 
 		size_t sharedMemorysize = reductions.product() * threads.x * threads.y * threads.z;
 
 		cudaMedianImageReduction<<<blocks,threads,sharedMemorysize>>>(*deviceImageIn, *deviceImageOut, reductions);

		//cudaMeanImageReduction<<<blocks,threads>>>(*deviceImageIn,*deviceImageOut,reductions);

		reducedIt->retriveROI(reducedImage,reducedDims,deviceImageOut);
		
		++orgIt;
		++reducedIt;
	}

	delete deviceImageIn;
	delete deviceImageOut;

	cudaThreadExit();

 	return reducedImage;
}

void CudaProcessBuffer::thresholdFilter(double threshold)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::unmix(const DevicePixelType* image, Vec<size_t> neighborhood)
{
	throw std::logic_error("The method or operation is not implemented.");
}

	// 	ImagePixelType otsuThresholdValue()
	// 	{
	// 		int temp;
	// 		return calcOtsuThreshold(retrieveNormalizedHistogram(temp),NUM_BINS);
	// 	}
	// 
	// 	ImagePixelType* retrieveImage(ImagePixelType* imageOut=NULL)
	// 	{
	// 		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
	// 		{
	// 			return NULL;
	// 		}
	// 		if (imageOut==NULL)
	// 			imageOut = new ImagePixelType[imageDims.product()];
	// 
	// 		const DevicePixelType* deviceImage = getCurrentBuffer()->getConstImagePointer();
	// 
	// 		HANDLE_ERROR(cudaMemcpy(imageOut,deviceImage,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToHost));
	// 		return imageOut;
	// 	}
	// 
	// 	void retrieveImage(ImageContainer* imageOut)
	// 	{
	// 		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
	// 		{
	// 			return;
	// 		}
	// 
	// 		HANDLE_ERROR(cudaMemcpy(imageOut->getMemoryPointer(),getCurrentBuffer(),sizeof(ImagePixelType)*imageDims.product(),
	// 			cudaMemcpyDeviceToHost));
	// 	}
	// 
	// 	/*
	// 	*	Returns a host pointer to the histogram data
	// 	*	This is destroyed when this' destructor is called
	// 	*	Will call the needed histogram creation methods if not all ready
	// 	*/
	// 	size_t* retrieveHistogram(int& returnSize)
	// 	{
	// 		if (!isCurrentNormHistogramHost)
	// 		{
	// 			createHistogram();
	// 
	// 			HANDLE_ERROR(cudaMemcpy(histogramHost,histogramDevice,sizeof(size_t)*NUM_BINS,cudaMemcpyDeviceToHost));
	// 			isCurrentHistogramHost = true;
	// 		}
	// 
	// 		returnSize = NUM_BINS;
	// 
	// 		return histogramHost;
	// 	}
	// 
	// 	/*
	// 	*	Returns a host pointer to the normalized histogram data
	// 	*	This is destroyed when this' destructor is called
	// 	*	Will call the needed histogram creation methods if not all ready
	// 	*/
	// 	double* retrieveNormalizedHistogram(int& returnSize)
	// 	{
	// 		if (!isCurrentNormHistogramHost)
	// 		{
	// 			normalizeHistogram();
	// 
	// 			HANDLE_ERROR(cudaMemcpy(normalizedHistogramHost,normalizedHistogramDevice,sizeof(double)*NUM_BINS,cudaMemcpyDeviceToHost));
	// 			isCurrentNormHistogramHost = true;
	// 		}
	// 
	// 		returnSize = NUM_BINS;
	// 
	// 		return normalizedHistogramHost;
	// 	}
	// 
	// 	ImagePixelType* retrieveReducedImage(Vec<size_t>& reducedDims)
	// 	{
	// 		reducedDims = this->reducedDims;
	// 
	// 		if (reducedImageDevice!=NULL)
	// 		{
	// 			HANDLE_ERROR(cudaMemcpy(reducedImageHost,reducedImageDevice,sizeof(ImagePixelType)*reducedDims.product(),cudaMemcpyDeviceToHost));
	// 		}
	// 
	// 		return reducedImageHost;
	// 	}
	// 
	// 	Vec<size_t> getDimension() const {return imageDims;}
	// 	int getDevice() const {return device;}
	// 	size_t getBufferSize() {return bufferSize;}
	// 
	// 	/*
	// 	*	This will replace this' cuda image buffer with the region of interest
	// 	*	from the passed in buffer.
	// 	*	****ENSURE that this' original size is big enough to accommodates the
	// 	*	the new buffer size.  Does not do error checking thus far.
	// 	*/
	// 	void copyROI(const CudaProcessBuffer<ImagePixelType>* image, Vec<size_t> starts, Vec<size_t> sizes)
	// 	{
	// 		if (sizes.product()>bufferSize || this->device!=image->getDevice())
	// 		{
	// 			clean();
	// 			this->device = image->getDevice();
	// 			imageDims = sizes;
	// 			deviceSetup();
	// 			memoryAllocation();
	// 		}
	// 
	// 		imageDims = sizes;
	// 		currentBuffer = 0;
	// 		image->getRoi(getCurrentBuffer(),starts,sizes);
	// 		updateBlockThread();
	// 	}
	// 
	// 	void copyROI(const CudaStorageBuffer<ImagePixelType>* imageIn, Vec<size_t> starts, Vec<size_t> sizes)
	// 	{
	// 		if ((size_t)sizes.product()>bufferSize || this->device!=imageIn->getDevice())
	// 		{
	// 			clean();
	// 			this->device = imageIn->getDevice();
	// 			imageDims = sizes;
	// 			deviceSetup();
	// 			memoryAllocation();
	// 		}
	// 
	// 		imageDims = sizes;
	// 		currentBuffer = 0;
	// 		imageIn->getRoi(getCurrentBuffer(),starts,sizes);
	// 		updateBlockThread();	
	// 	}
	// 
	// 	void copyImage(const CudaProcessBuffer<ImagePixelType>* bufferIn)
	// 	{
	// 		if (bufferIn->getDimension().product()>bufferSize)
	// 		{
	// 			clean();
	// 			this->device = device;
	// 			imageDims = bufferIn->getDimension();
	// 			deviceSetup();
	// 			memoryAllocation();
	// 		}
	// 
	// 		imageDims = bufferIn->getDimension();
	// 		device = bufferIn->getDevice();
	// 		updateBlockThread();
	// 
	// 		currentBuffer = 0;
	// 		HANDLE_ERROR(cudaMemcpy(getCurrentBuffer(),bufferIn->getCudaBuffer(),sizeof(ImagePixelType)*imageDims.product(),
	// 			cudaMemcpyDeviceToDevice));
	// 	}
	// 
	// 	const CudaImageContainer* getCudaBuffer() const
	// 	{
	// 		return getCurrentBuffer();
	// 	}
	// 
	// 	size_t getMemoryUsed() {return memoryUsage;}
	// 	size_t getGlobalMemoryAvailable() {return deviceProp.totalGlobalMem;}
	// 


//void memoryAllocation();
// 	{
// 		assert(sizeof(ImagePixelType)*imageDims.product()*NUM_BUFFERS < deviceProp.totalGlobalMem*.8);
// 
// 		for (int i=0; i<NUM_BUFFERS; ++i)
// 		{
// 			imageBuffers[i] = new CudaImageContainerClean(imageDims,device);
// 		}
// 
// 		currentBuffer = -1;
// 		bufferSize = imageDims.product();
// 
// 		updateBlockThread();
// 
// 		sizeSum = sumBlocks.x;
// 		HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*sumBlocks.x));
// 		memoryUsage += sizeof(double)*sumBlocks.x;
// 		hostSum = new double[sumBlocks.x];
// 
// 		HANDLE_ERROR(cudaMalloc((void**)&minValuesDevice,sizeof(double)*sumBlocks.x));
// 		memoryUsage += sizeof(double)*sumBlocks.x;
// 
// 		histogramHost = new size_t[NUM_BINS];
// 		HANDLE_ERROR(cudaMalloc((void**)&histogramDevice,NUM_BINS*sizeof(size_t)));
// 		memoryUsage += NUM_BINS*sizeof(size_t);
// 
// 		normalizedHistogramHost = new double[NUM_BINS];
// 		HANDLE_ERROR(cudaMalloc((void**)&normalizedHistogramDevice,NUM_BINS*sizeof(double)));
// 		memoryUsage += NUM_BINS*sizeof(double);
// 
// 		minPixel = std::numeric_limits<ImagePixelType>::min();
// 		maxPixel = std::numeric_limits<ImagePixelType>::max();
// 	}
// 
// 	void setStatus( Vec<size_t> dims )
// 	{
// 		if (dims.product()>bufferSize)
// 		{
// 			int device = this->device;
// 			clean();
// 			this->device = device;
// 			imageDims = dims;
// 			deviceSetup();
// 			memoryAllocation();
// 		}
// 		else
// 		{
// 			isCurrentHistogramHost = false;
// 			isCurrentHistogramDevice = false;
// 			isCurrentNormHistogramHost = false;
// 			isCurrentNormHistogramDevice = false;
// 		}
// 
// 		imageDims = dims;
// 		currentBuffer = 0;
// 		reservedBuffer = -1;
// 	}
// 
// 	void getRoi(ImagePixelType* roi, Vec<size_t> starts, Vec<size_t> sizes) const
// 	{
// #if CUDA_CALLS_ON
// 		cudaGetROI<<<blocks,threads>>>(*getCurrentBuffer(),roi,starts,sizes);
// #endif
// 	}
// 
// 	void copy(const CudaProcessBuffer<ImagePixelType>* bufferIn)
// 	{
// 		defaults();
// 
// 		imageDims = bufferIn->getDimension();
// 		device = bufferIn->getDevice();
// 
// 		deviceSetup();
// 		memoryAllocation();
// 
// 		currentBuffer = 0;
// 		ImagePixelType* inImage = bufferIn->getCurrentBuffer();
// 
// 		if (inImage!=NULL)
// 			HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],inImage,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToDevice));
// 
// 		if (bufferIn->reducedImageHost!=NULL)
// 			memcpy(reducedImageHost,bufferIn->reducedImageHost,sizeof(ImagePixelType)*reducedDims.product());
// 
// 		if (bufferIn->reducedImageDevice!=NULL)
// 			HANDLE_ERROR(cudaMemcpy(reducedImageDevice,bufferIn->reducedImageDevice,sizeof(ImagePixelType)*reducedDims.product(),
// 			cudaMemcpyDeviceToDevice));
// 
// 		if (bufferIn->histogramHost!=NULL)
// 			memcpy(histogramHost,bufferIn->histogramHost,sizeof(size_t)*imageDims.product());
// 
// 		if (bufferIn->histogramDevice!=NULL)
// 			HANDLE_ERROR(cudaMemcpy(histogramDevice,bufferIn->histogramDevice,sizeof(size_t)*NUM_BINS,cudaMemcpyDeviceToDevice));
// 
// 		if (bufferIn->normalizedHistogramHost!=NULL)
// 			memcpy(normalizedHistogramHost,bufferIn->normalizedHistogramHost,sizeof(double)*imageDims.product());
// 
// 		if (bufferIn->normalizedHistogramDevice!=NULL)
// 			HANDLE_ERROR(cudaMemcpy(normalizedHistogramDevice,bufferIn->normalizedHistogramDevice,sizeof(double)*NUM_BINS,
// 			cudaMemcpyDeviceToDevice));
// 	}
// 
// 	void constKernelOnes()
// 	{
// 		memset(hostKernel,1,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
// 		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM));
// 	}
// 
// 	void constKernelZeros()
// 	{
// 		memset(hostKernel,1,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
// 		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM));
// 	}
// 
// 	void setConstKernel(double* kernel, Vec<size_t> kernelDims)
// 	{
// 		memset(hostKernel,0,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
// 
// 		Vec<size_t> coordinate(0,0,0);
// 		for (; coordinate.x<kernelDims.x; ++coordinate.x)
// 		{
// 			coordinate.y = 0;
// 			for (; coordinate.y<kernelDims.y; ++coordinate.y)
// 			{
// 				coordinate.z = 0;
// 				for (; coordinate.z<kernelDims.z; ++coordinate.z)
// 				{
// 					hostKernel[kernelDims.linearAddressAt(coordinate)] = (float)kernel[kernelDims.linearAddressAt(coordinate)];
// 				}
// 			}
// 		}
// 		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*kernelDims.product()));
// 	}
// 
// 	void defaults()
// 	{
// 		imageDims = UNSET;
// 		reducedDims = UNSET;
// 		constKernelDims = UNSET;
// 		gausKernelSigmas  = Vec<float>(0.0f,0.0f,0.0f);
// 		device = -1;
// 		currentBuffer = -1;
// 		bufferSize = 0;
// 		for (int i=0; i<NUM_BUFFERS; ++i)
// 		{
// 			imageBuffers[i] = NULL;
// 		}
// 
// 		reducedImageHost = NULL;
// 		reducedImageDevice = NULL;
// 		histogramHost = NULL;
// 		histogramDevice = NULL;
// 		normalizedHistogramHost = NULL;
// 		normalizedHistogramDevice = NULL;
// 		isCurrentHistogramHost = false;
// 		isCurrentHistogramDevice = false;
// 		isCurrentNormHistogramHost = false;
// 		isCurrentNormHistogramDevice = false;
// 		deviceSum = NULL;
// 		minValuesDevice = NULL;
// 		hostSum = NULL;
// 		gaussIterations = Vec<int>(0,0,0);
// 		reservedBuffer = -1;
// 		memoryUsage = 0;
// 	}
// 
// 	void clean() 
// 	{
// 		for (int i=0; i<NUM_BUFFERS && imageBuffers!=NULL; ++i)
// 		{
// 			if (imageBuffers[i]!=NULL)
// 				delete imageBuffers[i];
// 		}
// 
// 		if (reducedImageHost!=NULL)
// 			delete reducedImageHost;
// 
// 		if (reducedImageDevice!=NULL)
// 			delete reducedImageDevice;
// 
// 		if (histogramHost!=NULL)
// 			delete[] histogramHost;
// 
// 		if (histogramDevice!=NULL)
// 			HANDLE_ERROR(cudaFree(histogramDevice));
// 
// 		if (normalizedHistogramHost!=NULL)
// 			delete[] normalizedHistogramHost;
// 
// 		if (normalizedHistogramDevice!=NULL)
// 			HANDLE_ERROR(cudaFree(normalizedHistogramDevice));
// 
// 		if (deviceSum!=NULL)
// 			HANDLE_ERROR(cudaFree(deviceSum));
// 
// 		if (hostSum!=NULL)
// 			delete[] hostSum;
// 
// 		if (minValuesDevice!=NULL)
// 			HANDLE_ERROR(cudaFree(minValuesDevice));
// 
// 		memset(hostKernel,0,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
// 
// 		defaults();
// 	}
// 
// 	CudaImageContainer* getCurrentBuffer() const 
// 	{
// 		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
// 			return NULL;
// 
// 		return imageBuffers[currentBuffer];
// 	}
// 
// 	CudaImageContainer* getNextBuffer()
// 	{
// 		return imageBuffers[getNextBufferNum()];
// 	}
// 
// 	int getNextBufferNum()
// 	{
// 		int nextIndex = currentBuffer;
// 		do 
// 		{
// 			++nextIndex;
// 			if (nextIndex>=NUM_BUFFERS)
// 				nextIndex = 0;
// 		} while (nextIndex==reservedBuffer);
// 		return nextIndex;
// 	}
// 
// 	CudaImageContainer* getReservedBuffer()
// 	{
// 		if (reservedBuffer<0)
// 			return NULL;
// 
// 		return imageBuffers[reservedBuffer];
// 	}
// 
// 	void reserveCurrentBuffer()
// 	{
// 		reservedBuffer = currentBuffer;
// 	}
// 
// 	void releaseReservedBuffer()
// 	{
// 		reservedBuffer = -1;
// 	}
// 
// 	void incrementBufferNumber()
// 	{
// 		cudaThreadSynchronize();
// #ifdef _DEBUG
// 		gpuErrchk( cudaPeekAtLastError() );
// #endif // _DEBUG
// 
// 		currentBuffer = getNextBufferNum();
// 	}