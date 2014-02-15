#include "CudaProcessBuffer.cuh"
#include "CudaUtilities.cuh"
#include "CudaStorageBuffer.cuh"
#include "CudaKernels.cuh"

CudaProcessBuffer::CudaProcessBuffer(int device/*=0*/)
{
	defaults();
	hostImageBuffers = NULL;
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

void CudaProcessBuffer::calculateChunking(Vec<size_t> kernalDims)
{
	if (orgImageDims==deviceDims)
		numChunks = Vec<size_t>(1,1,1);
	else
	{
		numChunks.x = (size_t)ceil((double)orgImageDims.x/(deviceDims.x-(kernalDims.x)));
		numChunks.y = (size_t)ceil((double)orgImageDims.y/(deviceDims.y-(kernalDims.y)));
		numChunks.z = (size_t)ceil((double)orgImageDims.z/(deviceDims.z-(kernalDims.z)));
	}

	hostImageBuffers = new ImageChunk[numChunks.product()];

	Vec<size_t> curChunk(0,0,0);
	Vec<size_t> curBuffStart(0,0,0);
	Vec<size_t> curImageStart(0,0,0);
	Vec<size_t> curBuffEnd(0,0,0);
	Vec<size_t> curImageEnd(0,0,0);
	Vec<size_t> curBuffSize(0,0,0);
	for (curChunk.z=0; curChunk.z<numChunks.z; ++curChunk.z)
	{
		if (curChunk.z==0)//first chunk
		{
			curBuffStart.z = 0;
			curImageStart.z = 0;
		}
		else
		{
			curBuffStart.z = 
				hostImageBuffers[numChunks.linearAddressAt(Vec<size_t>(curChunk.x,curChunk.y,curChunk.z-1))].endImageIdx.z-kernalDims.z;

			curImageStart.z = hostImageBuffers[numChunks.linearAddressAt(Vec<size_t>(curChunk.x,curChunk.y,curChunk.z-1))].endImageIdx.z +1;
		}

		curBuffSize.z = min(deviceDims.z, orgImageDims.z-curBuffStart.z);

		if (curBuffSize.z<=deviceDims.z) //last chunk
		{
			curBuffEnd.z = orgImageDims.z;
			curImageEnd.z = orgImageDims.z;
		}
		else
		{
			curBuffEnd.z = min(orgImageDims.z,curBuffStart.z+deviceDims.z);
			curImageEnd.z = min(orgImageDims.z,curBuffStart.z+deviceDims.z-kernalDims.z);
		}

		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
		{
			if (curChunk.y==0)//first chunk
			{
				curBuffStart.y = 0;
				curImageStart.y = 0;
			}
			else
			{
				curBuffStart.y = 
					hostImageBuffers[numChunks.linearAddressAt(Vec<size_t>(curChunk.x,curChunk.y-1,curChunk.z))].endImageIdx.y-kernalDims.y;

				curImageStart.y = hostImageBuffers[numChunks.linearAddressAt(Vec<size_t>(curChunk.x,curChunk.y-1,curChunk.z))].endImageIdx.y +1;
			}

			curBuffSize.y = min(deviceDims.y, orgImageDims.y-curBuffStart.y);

			if (curBuffSize.y<=deviceDims.y) //last chunk
			{
				curBuffEnd.y = orgImageDims.y;
				curImageEnd.y = orgImageDims.y;
			}
			else
			{
				curBuffEnd.y = min(orgImageDims.y,curBuffStart.y+deviceDims.y);
				curImageEnd.y = min(orgImageDims.y,curBuffStart.y+deviceDims.y-kernalDims.y);
			}

			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
			{
				if (curChunk.x==0)//first chunk
				{
					curBuffStart.x = 0;
					curImageStart.x = 0;
				}
				else
				{
					curBuffStart.x = 
						hostImageBuffers[numChunks.linearAddressAt(Vec<size_t>(curChunk.x-1,curChunk.y,curChunk.z))].endImageIdx.x-kernalDims.x;

					curImageStart.x = hostImageBuffers[numChunks.linearAddressAt(Vec<size_t>(curChunk.x-1,curChunk.y,curChunk.z))].endImageIdx.x +1;
				}

				curBuffSize.x = min(deviceDims.x, orgImageDims.x-curBuffStart.x);

				if (curBuffSize.x<=deviceDims.x) //last chunk
				{
					curBuffEnd.x = orgImageDims.x;
					curImageEnd.x = orgImageDims.x;
				}
				else
				{
					curBuffEnd.x = min(orgImageDims.x,curBuffStart.x+deviceDims.x);
					curImageEnd.x = min(orgImageDims.x,curBuffStart.x+deviceDims.x-kernalDims.x);
				}

				ImageChunk* curImageBuffer = &hostImageBuffers[numChunks.linearAddressAt(curChunk)];

				curImageBuffer->image = new ImageContainer(curBuffSize);

				curImageBuffer->startImageIdx = curImageStart;
				curImageBuffer->startBuffIdx = curBuffStart;
				curImageBuffer->endImageIdx = curImageEnd;
				curImageBuffer->endBuffIdx = curBuffEnd;
			}

			curChunk.x = 0;
		}

		curChunk.y = 0;
	}
}

void CudaProcessBuffer::deviceSetup()
{
	HANDLE_ERROR(cudaSetDevice(device));
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
}

void CudaProcessBuffer::createDeviceBuffers(int numBuffersNeeded, Vec<size_t> kernalDims)
{
	clearDeviceBuffers();
	deviceImageBuffers.resize(numBuffersNeeded);

	size_t numVoxels = (size_t)((double)deviceProp.totalGlobalMem*0.9/(sizeof(HostPixelType)*numBuffersNeeded));

	Vec<size_t> dimWkernal(orgImageDims+kernalDims);
	if (dimWkernal>orgImageDims)
		deviceDims = orgImageDims;
	else
	{
		deviceDims = Vec<size_t>(0,0,orgImageDims.z);
		double leftOver = (double)numVoxels/dimWkernal.z;

		double squareDim = sqrt(leftOver);
		if (squareDim>dimWkernal.y)
		{
			deviceDims.y = dimWkernal.y;
			deviceDims.x = (size_t)(leftOver/dimWkernal.y);
			if (deviceDims.x>dimWkernal.x)
				deviceDims.x = dimWkernal.x;
		}
		else 
		{
			deviceDims.x = (size_t)squareDim;
			deviceDims.y = (size_t)squareDim;
		}
	}

	for (int i=0; i<numBuffersNeeded; ++i)
	{
		deviceImageBuffers[i] = new CudaImageContainerClean(deviceDims,device);
	}

	currentBufferIdx = 0;
	updateBlockThread();
	calculateChunking(kernalDims);
}

void CudaProcessBuffer::updateBlockThread()
{
	calcBlockThread(deviceDims,deviceProp,blocks,threads);
}

void CudaProcessBuffer::defaults()
{
	device = 0;
	blocks = dim3(0,0,0);
	threads = dim3(0,0,0);
	orgImageDims = Vec<size_t>(0,0,0);
	numChunks = Vec<size_t>(0,0,0);
	curChunkIdx = Vec<size_t>(0,0,0);
	currentBufferIdx = -1;
	deviceDims = Vec<size_t>(0,0,0);
}

void CudaProcessBuffer::createBuffers()
{
	throw std::logic_error("The method or operation is not implemented.");
// 	hostImageBuffers = new ImageContainer*[numChunks.product()];
// 	Vec<size_t> curChunk(0,0,0);
// 	Vec<size_t> startIdx(0,0,0);
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
			if (hostImageBuffers[i].image!=NULL)
			{
				delete hostImageBuffers[i].image;
				hostImageBuffers[i].image = NULL;
			}
		}

		delete[] hostImageBuffers;
		hostImageBuffers = NULL;
	}

	clearDeviceBuffers();
}

void CudaProcessBuffer::loadImage(HostPixelType* imageIn)
{
	throw std::logic_error("The method or operation is not implemented.");
// 	if (numChunks.product() == 1)
// 	{
// 		hostImageBuffers[0]->setImagePointer(imageIn,orgImageDims);
// 		return;
// 	}
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

void CudaProcessBuffer::incrementBufferNumber()
{
	++currentBufferIdx;
	if (currentBufferIdx>=deviceImageBuffers.size())
		currentBufferIdx = 0;
}

CudaImageContainer* CudaProcessBuffer::getCurrentBuffer()
{
	return deviceImageBuffers[currentBufferIdx];
}

CudaImageContainer* CudaProcessBuffer::getNextBuffer()
{
	int nextIdx = (currentBufferIdx+1 >= deviceImageBuffers.size()) ? 0 : currentBufferIdx+1;
	return deviceImageBuffers[nextIdx];
}

bool CudaProcessBuffer::loadNextChunk(const DevicePixelType* imageIn)
{
	static bool finished = false;

	if (finished)
	{
		finished = false;
		curChunkIdx = Vec<size_t>(0,0,0);
		return false;
	}

	if (numChunks.product()==1)
	{
		getCurrentBuffer()->loadImage(imageIn,orgImageDims);
		finished = true;
		return true;
	}

	ImageChunk curChunk = hostImageBuffers[numChunks.linearAddressAt(curChunkIdx)];
	Vec<size_t> curHostIdx(curChunk.startBuffIdx);
	Vec<size_t> curDeviceIdx(0,0,0);
	Vec<size_t> curBuffSize = curChunk.endBuffIdx-curChunk.startBuffIdx;
	DevicePixelType* deviceImage = getCurrentBuffer()->getImagePointer();

	for (curHostIdx.z=curChunk.startBuffIdx.z; curHostIdx.z<curChunk.endBuffIdx.z; ++curHostIdx.z)
	{
		curDeviceIdx.y = 0;
		for (curHostIdx.y=curChunk.startBuffIdx.y; curHostIdx.y<curChunk.endBuffIdx.y; ++curHostIdx.y)
		{
			HANDLE_ERROR(cudaMemcpy(deviceImage+curDeviceIdx.y*curBuffSize.x+curDeviceIdx.z*curBuffSize.y*curBuffSize.x,
				imageIn+curHostIdx.product(),sizeof(DevicePixelType)*curBuffSize.x,cudaMemcpyHostToDevice));

			++curDeviceIdx.y;
		}
		++curDeviceIdx.z;
	}

	++curChunkIdx.x;
	if (curChunkIdx.x>=numChunks.x)
	{
		curChunkIdx.x = 0;
		++curChunkIdx.y;

		if (curChunkIdx.y>=numChunks.y)
		{
			curChunkIdx.y = 0;
			++curChunkIdx.z;

			if (curChunkIdx.z>=numChunks.z)
			{
				finished = true;
			}
		}
	}

	return true;
}

void CudaProcessBuffer::saveCurChunk(DevicePixelType* imageOut)
{
	const DevicePixelType* deviceImage = getCurrentBuffer()->getConstImagePointer();
	if (numChunks.product()==1)
	{
		HANDLE_ERROR(cudaMemcpy(imageOut,deviceImage,sizeof(DevicePixelType)*orgImageDims.product(),cudaMemcpyDeviceToHost));
	}
	else
	{
		ImageChunk curChunk = hostImageBuffers[numChunks.linearAddressAt(curChunkIdx)];
		Vec<size_t> curHostIdx(curChunk.startImageIdx);
		Vec<size_t> curDeviceIdx(0,0,0);

		for (curHostIdx.z=curChunk.startImageIdx.z; curHostIdx.z<curChunk.endImageIdx.z; ++curHostIdx.z)
		{
			for (curHostIdx.y=curChunk.startImageIdx.y; curHostIdx.y<curChunk.endImageIdx.y; ++curHostIdx.y)
			{
				HANDLE_ERROR(cudaMemcpy(imageOut+curHostIdx.product(),deviceImage+(curHostIdx-curChunk.startBuffIdx).product(),
					sizeof(DevicePixelType)*(curChunk.endImageIdx.x-curChunk.startImageIdx.x),cudaMemcpyDeviceToHost));
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

void CudaProcessBuffer::addConstant(double additive)
{
	throw std::logic_error("The method or operation is not implemented.");
}

void CudaProcessBuffer::addImageWith(const DevicePixelType* image, double factor)
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
	return NULL;
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
	
	createDeviceBuffers(2,neighborhood);

	while (loadNextChunk(imageIn))
	{
		cudaMeanFilter<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),neighborhood);
		saveCurChunk(meanImage);
	}

	return meanImage;
}

void CudaProcessBuffer::medianFilter(Vec<size_t> neighborhood)
{
	throw std::logic_error("The method or operation is not implemented.");
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
	return 0.0;
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

HostPixelType* CudaProcessBuffer::reduceImage(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims)
{
	throw std::logic_error("The method or operation is not implemented.");
// 	orgImageDims = dims;
// 	Vec<size_t> boarder((size_t)ceil(reductions.x/2.0), (size_t)ceil(reductions.y/2.0), (size_t)ceil(reductions.z/2.0));
// 	createDeviceBuffers(2,boarder);
// 	reducedDims.x = (size_t)ceil(dims.x/reductions.x);
// 	reducedDims.y = (size_t)ceil(dims.y/reductions.y);
// 	reducedDims.z = (size_t)ceil(dims.z/reductions.z);
// 
// 	HostPixelType* outImage = new HostPixelType[reducedDims.product()];
// 
// 	if (numChunks.product()==1)//image fits on the device
// 		deviceImageBuffers[0]->loadImage(imageIn,dims);
// 	else
// 	{
// 		loadNextChunk(imageIn);
// 		cudaRuduceImage<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),reductions);
// 
// 		HANDLE_ERROR(cudaMemcpy(curChunk.image->getMemoryPointer(),getNextBuffer()->getDeviceImagePointer(),
// 			sizeof(DevicePixelType)*curBuffSize.product(),cudaMemcpyDeviceToHost));
// 
// 	}

	return NULL;
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