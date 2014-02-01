#include "CudaProcessBuffer.cuh"
#include "CudaUtilities.cuh"
#include "CudaStorageBuffer.cuh"
#include "CudaKernels.cuh"

CudaProcessBuffer::CudaProcessBuffer(int device/*=0*/)
{
	defaults();
	hostImageBuffers = NULL;
	deviceImageBuffers = NULL;
	this->device = device;
	deviceSetup();
}

// CudaProcessBuffer::CudaProcessBuffer(HostPixelType* imageIn, Vec<size_t> dims, int device/*=0*/)
// {
// 	defaults();
// 
// 	orgImageDims = dims;
// 	this->device = device;
// 
// 	calculateChunking();
// 	createBuffers();
// 
// 	loadImage(imageIn);
// }

CudaProcessBuffer::~CudaProcessBuffer()
{
	clearBuffers();
	defaults();
}

void CudaProcessBuffer::calculateChunking()
{
	size_t numVoxels = (size_t)(((double)deviceProp.totalGlobalMem/sizeof(HostPixelType))*0.4);

	if (orgImageDims.z==1)
	{
		if (orgImageDims.y==1)
		{
			if (orgImageDims.x<numVoxels)
			{
				chunkDims.x = orgImageDims.x;
				chunkDims.y = 1;
				chunkDims.z = 1;

				numChunks.x = 1;
				numChunks.y = 1;
				numChunks.z = 1;
			} 
			else
			{
				chunkDims.x = numVoxels;
				chunkDims.y = 1;
				chunkDims.z = 1;

				numChunks.x = (int)ceil((float)orgImageDims.x/numVoxels);
				numChunks.y = 1;
				numChunks.z = 1;
			}
		}
		else
		{
			if (orgImageDims.x*orgImageDims.y<numVoxels)
			{
				chunkDims.x = orgImageDims.x;
				chunkDims.y = orgImageDims.y;
				chunkDims.z = 1;

				numChunks.x = 1;
				numChunks.y = 1;
				numChunks.z = 1;
			} 
			else
			{
				int dim = (int)sqrt((double)numVoxels);

				chunkDims.x = dim;
				chunkDims.y = dim;
				chunkDims.z = 1;

				numChunks.x = (int)ceil((float)orgImageDims.x/dim);
				numChunks.y = (int)ceil((float)orgImageDims.y/dim);
				numChunks.z = 1;
			}
		}
	}
	else
	{
		if(orgImageDims.product()<numVoxels)
		{
			chunkDims.x = orgImageDims.x;
			chunkDims.y = orgImageDims.y;
			chunkDims.z = orgImageDims.z;

			numChunks.x = 1;
			numChunks.y = 1;
			numChunks.z = 1;
		}
		else
		{
			Vec<size_t> dims;
			size_t dim = (size_t)pow((float)numVoxels,1/3.0f);
			if (dim>orgImageDims.z)
			{
				dim = (int)sqrt((double)numVoxels/orgImageDims.z);
				dims.z = orgImageDims.z;
				dims.x = dim;
				dims.y = dim;
			}

			float extra = (float)(numVoxels-dims.x*dims.y*dims.z)/(dims.x*dims.y);

			chunkDims.x = dims.x + (int)extra;
			chunkDims.y = dims.y;
			chunkDims.z = dims.z;

			numChunks.x = (size_t)ceil((float)orgImageDims.x/chunkDims.x);
			numChunks.y = (size_t)ceil((float)orgImageDims.y/chunkDims.y);
			numChunks.z = (size_t)ceil((float)orgImageDims.z/chunkDims.z);
		}
	}
}

void CudaProcessBuffer::deviceSetup()
{
	HANDLE_ERROR(cudaSetDevice(device));
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
	calculateChunking();
	updateBlockThread();
}

void CudaProcessBuffer::updateBlockThread()
{
	calcBlockThread(deviceDims,deviceProp,blocks,threads);
}

void CudaProcessBuffer::defaults()
{
	bufferSize = 0;
	device = 0;
	blocks = dim3(0,0,0);
	threads = dim3(0,0,0);
	orgImageDims = Vec<size_t>(0,0,0);
	chunkDims = Vec<size_t>(0,0,0);
	numChunks = Vec<size_t>(0,0,0);
	numDeviceBuffers = 0;
	deviceDims = Vec<size_t>(0,0,0);
}

void CudaProcessBuffer::createBuffers()
{
	hostImageBuffers = new ImageContainer*[numChunks.product()];
	Vec<size_t> curChunk(0,0,0);
	Vec<size_t> startIdx(0,0,0);
	Vec<size_t> curChunkDim;
// 	for (; curChunk.z<numChunks.z; ++curChunk.z)
// 	{
// 		if (curChunk.z*chunkDims.z+chunkDims.z<orgImageDims.z)
// 		{
// 			curChunkDim.z = chunkDims.z;
// 		}
// 		else
// 		{
// 			curChunkDim.z = orgImageDims.z - curChunk.z*chunkDims.z;
// 		}
// 
// 		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
// 		{
// 			if (curChunk.y*chunkDims.y+chunkDims.y<orgImageDims.y)
// 			{
// 				curChunkDim.y = chunkDims.y;
// 			}
// 			else
// 			{
// 				curChunkDim.y = orgImageDims.y - curChunk.y*chunkDims.y;
// 			}
// 
// 			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
// 			{
// 				if (curChunk.x*chunkDims.x+chunkDims.x<orgImageDims.x)
// 				{
// 					curChunkDim.x = chunkDims.x;
// 				}
// 				else
// 				{
// 					curChunkDim.x = orgImageDims.x - curChunk.x*chunkDims.x;
// 				}
// 
// 				hostImageBuffers[numChunks.linearAddressAt(curChunk)] = new ImageContainer(curChunkDim);
// 			}
// 		}
// 	}

// 	for (int i=0; i<NUM_DEVICE_BUFFERS; ++i)
// 		deviceImageBuffers[i] = new CudaImageContainerClean(chunkDims,device);
}

void CudaProcessBuffer::clearBuffers()
{
	if (hostImageBuffers!=NULL)
	{
		for (int i=0; i<numChunks.product(); ++i)
		{
			if (hostImageBuffers[i]!=NULL)
			{
				delete hostImageBuffers[i];
				hostImageBuffers[i] = NULL;
			}
		}

		delete[] hostImageBuffers;
		hostImageBuffers = NULL;
	}

	if (deviceImageBuffers!=NULL)
	{
		for (int i=0; i<numDeviceBuffers; ++i)
		{
			if (deviceImageBuffers[i]!=NULL)
			{
				delete deviceImageBuffers[i];
				deviceImageBuffers[i] = NULL;
			}
		}
		delete[] deviceImageBuffers;
		deviceImageBuffers = NULL;
	}
}

void CudaProcessBuffer::loadImage(HostPixelType* imageIn)
{
	if (numChunks.product() == 1)
	{
		hostImageBuffers[0]->setImagePointer(imageIn,orgImageDims);
		return;
	}
// 
// 	Vec<size_t> curChunk(0,0,0);
// 	Vec<size_t> curChunkDim;
// 	for (; curChunk.z<numChunks.z; ++curChunk.z)
// 	{
// 		if (curChunk.z*chunkDims.z+chunkDims.z<orgImageDims.z)
// 		{
// 			curChunkDim.z = chunkDims.z;
// 		}
// 		else
// 		{
// 			curChunkDim.z = orgImageDims.z - curChunk.z*chunkDims.z;
// 		}
// 
// 		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
// 		{
// 			if (curChunk.y*chunkDims.y+chunkDims.y<orgImageDims.y)
// 			{
// 				curChunkDim.y = chunkDims.y;
// 			}
// 			else
// 			{
// 				curChunkDim.y = orgImageDims.y - curChunk.y*chunkDims.y;
// 			}
// 
// 			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
// 			{
// 				if (curChunk.x*chunkDims.x+chunkDims.x<orgImageDims.x)
// 				{
// 					curChunkDim.x = chunkDims.x;
// 				}
// 				else
// 				{
// 					curChunkDim.x = orgImageDims.x - curChunk.x*chunkDims.x;
// 				}
// 
// 				HostPixelType* im = hostImageBuffers[numChunks.linearAddressAt(curChunk)]->getMemoryPointer();
// 				Vec<size_t> startIdx(chunkDims.x*curChunk.x,chunkDims.y*curChunk.y,chunkDims.z*curChunk.z);
// 				Vec<size_t> curIdx(startIdx);
// 
// 				for (; curIdx.z<startIdx.z+chunkDims.z && curIdx.z<curChunkDim.z; ++curIdx.z)
// 				{
// 					for(curIdx.y=startIdx.y; curIdx.y<startIdx.y+chunkDims.y && curIdx.y<curChunkDim.y; ++curIdx.y)
// 					{
// 						memcpy(im+chunkDims.linearAddressAt(curIdx-startIdx),imageIn+orgImageDims.linearAddressAt(curIdx),
// 							sizeof(HostPixelType)*curChunkDim.x);
// 					}
// 				}
// 			}
// 		}
// 	}
}

void CudaProcessBuffer::addConstant(double additive)
{

}

void CudaProcessBuffer::addImageWith(const DevicePixelType* image, double factor)
{

}

void CudaProcessBuffer::applyPolyTransformation(double a, double b, double c, DevicePixelType minValue, DevicePixelType maxValue)
{

}

void CudaProcessBuffer::calculateMinMax(double& minValue, double& maxValue)
{

}

void CudaProcessBuffer::contrastEnhancement(Vec<float> sigmas, Vec<size_t> medianNeighborhood)
{

}

void CudaProcessBuffer::createHistogram()
{

}

void CudaProcessBuffer::gaussianFilter(Vec<float> sigmas)
{

}

void CudaProcessBuffer::mask(const DevicePixelType* imageMask, DevicePixelType threshold/*=1*/)
{

}

void CudaProcessBuffer::maxFilter(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{

}

void CudaProcessBuffer::maximumIntensityProjection()
{

}

DevicePixelType* CudaProcessBuffer::meanFilter(DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood,
											 DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* meanImage;
	
	createDeviceBuffers(dims, 2);
	
	if (dims==deviceDims)
		deviceImageBuffers[0]->loadImage(imageIn,dims);
	else
		;

	if (imageOut==NULL)
		meanImage = new DevicePixelType[deviceDims.product()];
	else
		meanImage = *imageOut;

	cudaMeanFilter<<<blocks,threads>>>(*(deviceImageBuffers[0]),*(deviceImageBuffers[1]),neighborhood);

	HANDLE_ERROR(cudaMemcpy(meanImage,deviceImageBuffers[1]->getDeviceImagePointer(),sizeof(DevicePixelType)*deviceDims.product(),cudaMemcpyDeviceToHost));

	return meanImage;
}

void CudaProcessBuffer::medianFilter(Vec<size_t> neighborhood)
{

}

void CudaProcessBuffer::minFilter(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{

}

void CudaProcessBuffer::morphClosure(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{

}

void CudaProcessBuffer::morphOpening(Vec<size_t> neighborhood, double* kernel/*=NULL*/)
{

}

void CudaProcessBuffer::multiplyImage(double factor)
{

}

void CudaProcessBuffer::multiplyImageWith(const DevicePixelType* image)
{

}

double CudaProcessBuffer::normalizedCovariance(DevicePixelType* otherImage)
{
	return 0.0;
}

void CudaProcessBuffer::normalizeHistogram()
{

}

void CudaProcessBuffer::otsuThresholdFilter(float alpha/*=1.0f*/)
{

}

void CudaProcessBuffer::imagePow(int p)
{

}

void CudaProcessBuffer::sumArray(double& sum)
{

}

void CudaProcessBuffer::reduceImage(Vec<double> reductions)
{

}

void CudaProcessBuffer::thresholdFilter(double threshold)
{

}

void CudaProcessBuffer::unmix(const DevicePixelType* image, Vec<size_t> neighborhood)
{

}

void CudaProcessBuffer::createDeviceBuffers(Vec<size_t> dims, int numBuffersNeeded)
{
	deviceImageBuffers = new CudaImageContainerClean*[numBuffersNeeded];
	numDeviceBuffers = numBuffersNeeded;

	size_t numVoxels = (size_t)((double)deviceProp.totalGlobalMem*0.9/(sizeof(HostPixelType)*numBuffersNeeded));
	deviceDims = Vec<size_t>(0,0,dims.z);
	double leftOver = (double)numVoxels/dims.z;

	double squareDim = sqrt(leftOver);
	if (squareDim>dims.y)
	{
		deviceDims.y = dims.y;
		deviceDims.x = (size_t)(leftOver/dims.y);
		if (deviceDims.x>dims.x)
			deviceDims.x = dims.x;
	}
	else 
	{
		deviceDims.x = (size_t)squareDim;
		deviceDims.y = (size_t)squareDim;
	}

	for (int i=0; i<numBuffersNeeded; ++i)
	{
		deviceImageBuffers[i] = new CudaImageContainerClean(deviceDims,device);
	}

	updateBlockThread();
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