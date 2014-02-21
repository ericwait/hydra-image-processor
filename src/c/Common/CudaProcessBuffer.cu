#include "CudaProcessBuffer.cuh"
#include "CudaUtilities.cuh"
#include "CudaStorageBuffer.cuh"
#include "CudaKernels.cuh"

CudaProcessBuffer::CudaProcessBuffer(int device/*=0*/)
{
	defaults();
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
	clearHostBuffers();
	clearDeviceBuffers();
	defaults();
}

void CudaProcessBuffer::calculateChunking(Vec<size_t> kernalDims)
{
	Vec<size_t> margin((kernalDims + 1)/2); //integer round
	Vec<size_t> chunkDelta(deviceDims-margin*2);
	numChunks = Vec<size_t>(1,1,1);

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

	hostImageBuffers.resize(numChunks.product());

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

				ImageChunk* curImageBuffer = &hostImageBuffers[numChunks.linearAddressAt(curChunk)];

				curImageBuffer->image = new ImageContainer(imageEnd-imageStart);

				curImageBuffer->imageStart = imageStart;
				curImageBuffer->chunkROIstart = chunkROIstart;
				curImageBuffer->imageROIstart = imageROIstart;
				curImageBuffer->imageEnd = imageEnd;
				curImageBuffer->chunkROIend = chunkROIend;
				curImageBuffer->imageROIend = imageROIend;
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

void CudaProcessBuffer::createDeviceBuffers(int numBuffersNeeded, Vec<size_t> kernalDims/*=Vec<size_t>(0,0,0)*/)
{
	clearDeviceBuffers();
	deviceImageBuffers.resize(numBuffersNeeded);

	size_t numVoxels = (size_t)((double)deviceProp.totalGlobalMem*0.9/(sizeof(HostPixelType)*numBuffersNeeded));

	deviceDims = Vec<size_t>(0,0,orgImageDims.z);
	double leftOver = (double)numVoxels/orgImageDims.z;

	double squareDim = sqrt(leftOver);

	if (squareDim>orgImageDims.y)
		deviceDims.y = orgImageDims.y;
	else 
		deviceDims.y = (size_t)squareDim;

	deviceDims.x = (size_t)(leftOver/deviceDims.y);

	if (deviceDims.x>orgImageDims.x)
		deviceDims.x = orgImageDims.x;

	for (int i=0; i<numBuffersNeeded; ++i)
		deviceImageBuffers[i] = new CudaImageContainerClean(deviceDims,device);

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
	nextChunkIdx = Vec<size_t>(0,0,0);
	lastChunk = false;
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

void CudaProcessBuffer::clearHostBuffers()
{
	if (!hostImageBuffers.empty())
	{
		for (std::vector<ImageChunk>::iterator it=hostImageBuffers.begin(); it!=hostImageBuffers.end(); ++it)
		{
			if (it->image!=NULL)
			{
				delete it->image;
				it->image = NULL;
			}
		}

		hostImageBuffers.clear();
	}
}

void CudaProcessBuffer::clearDeviceBuffers()
{
	if (!deviceImageBuffers.empty())
	{
		for (std::vector<CudaImageContainerClean*>::iterator it=deviceImageBuffers.begin(); it!=deviceImageBuffers.end(); ++it)
		{
			if (*it!=NULL)
			{
				delete *it;
				*it = NULL;
			}
		}

		deviceImageBuffers.clear();
	}
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
	curChunkIdx = nextChunkIdx;

	if (lastChunk)
	{
		lastChunk = false;
		return false;
	}

	if (numChunks.product()==1)
	{
		getCurrentBuffer()->loadImage(imageIn,orgImageDims);
		lastChunk = true;
		return true;
	}

	ImageChunk* curChunk = &hostImageBuffers[numChunks.linearAddressAt(curChunkIdx)];
	Vec<size_t> curIdx(0,0,0);
	Vec<size_t> curChunkSize = curChunk->imageEnd - curChunk->imageStart;

	if (!getCurrentBuffer()->setDims(curChunkSize))
		throw std::runtime_error("Unable to load chunk to the device because the buffer was too small!");

	for (curIdx.z=0; curIdx.z<curChunkSize.z; ++curIdx.z)
	{
		for (curIdx.y=0; curIdx.y<curChunkSize.y; ++curIdx.y)
		{
			Vec<size_t> curHostIdx = curChunk->imageStart + curIdx;
			Vec<size_t> curDeviceIdx = curIdx;

			const DevicePixelType* hostPtr = imageIn + orgImageDims.linearAddressAt(curHostIdx);
			//DevicePixelType* devicePtr = curChunk->image->getMemoryPointer() + curChunkSize.linearAddressAt(curDeviceIdx);
			DevicePixelType* devicePtr = getCurrentBuffer()->getImagePointer() + curChunkSize.linearAddressAt(curDeviceIdx);

			//memcpy(devicePtr,hostPtr,sizeof(DevicePixelType)*curChunkSize.x);
			HANDLE_ERROR(cudaMemcpy(devicePtr,hostPtr,sizeof(DevicePixelType)*curChunkSize.x,cudaMemcpyHostToDevice));
		}
	}

	++nextChunkIdx.x;
	if (nextChunkIdx.x>=numChunks.x)
	{
		nextChunkIdx.x = 0;
		++nextChunkIdx.y;

		if (nextChunkIdx.y>=numChunks.y)
		{
			nextChunkIdx.y = 0;
			++nextChunkIdx.z;

			if (nextChunkIdx.z>=numChunks.z)
			{
				lastChunk = true;
				nextChunkIdx = Vec<size_t>(0,0,0);
			}
		}
	}

	return true;
}

void CudaProcessBuffer::retriveCurChunk()
{
	ImageChunk* curChunk = &hostImageBuffers[numChunks.linearAddressAt(curChunkIdx)];

	Vec<size_t> curBuffSize = curChunk->imageEnd - curChunk->imageStart;

 	HANDLE_ERROR(cudaMemcpy(curChunk->image->getMemoryPointer(), getCurrentBuffer()->getConstImagePointer(),
 		sizeof(DevicePixelType)*curBuffSize.product(), cudaMemcpyDeviceToHost));
}

void CudaProcessBuffer::saveChunks(DevicePixelType* imageOut)
{
	if (numChunks.product()==1)
	{
		ImageChunk* curChunk = &hostImageBuffers[numChunks.linearAddressAt(curChunkIdx)];
		memcpy(imageOut,hostImageBuffers[0].image->getConstMemoryPointer(),sizeof(DevicePixelType)*orgImageDims.product());
	}
	else
	{
		Vec<size_t> localChunkIdx(0,0,0);
		for (localChunkIdx.z=0; localChunkIdx.z<numChunks.z; ++localChunkIdx.z)
		{
			for (localChunkIdx.y=0; localChunkIdx.y<numChunks.y; ++localChunkIdx.y)
			{
				for (localChunkIdx.x=0; localChunkIdx.x<numChunks.x; ++localChunkIdx.x)
				{
					ImageChunk* curChunk = &hostImageBuffers[numChunks.linearAddressAt(localChunkIdx)];
					Vec<size_t> roiIdx = Vec<size_t>(0,0,0);
					Vec<size_t> curChunkSize = curChunk->imageEnd - curChunk->imageStart;
					Vec<size_t> curROISize = curChunk->imageROIend - curChunk->imageROIstart;

					const DevicePixelType* chunkImagePtr = curChunk->image->getConstMemoryPointer();

					for (roiIdx.z=0; roiIdx.z<curROISize.z; ++roiIdx.z)
					{
						for (roiIdx.y=0; roiIdx.y<curROISize.y; ++roiIdx.y)
						{
							Vec<size_t> chunkIdx(curChunk->chunkROIstart+roiIdx);
							Vec<size_t> outIdx(curChunk->imageROIstart+roiIdx);

							DevicePixelType* outPtr = imageOut + orgImageDims.linearAddressAt(outIdx);
							const DevicePixelType* chunkPtr = chunkImagePtr + curChunkSize.linearAddressAt(chunkIdx);

							memcpy(outPtr,chunkPtr,sizeof(DevicePixelType)*curROISize.x);
						}
					}
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

DevicePixelType* CudaProcessBuffer::addConstant(const DevicePixelType* imageIn, Vec<size_t> dims, double additive, DevicePixelType** imageOut/*=NULL*/)
{
	orgImageDims = dims;

	DevicePixelType* imOut;
	if (imageOut==NULL)
		imOut = new DevicePixelType[orgImageDims.product()];
	else
		imOut = *imageOut;

	createDeviceBuffers(2);

	while (loadNextChunk(imageIn))
	{
		if (!getNextBuffer()->setDims(getCurrentBuffer()->getDims()))
			throw std::runtime_error("Unable to load chunk to the device because the buffer was too small!");

		cudaAddFactor<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),additive,std::numeric_limits<DevicePixelType>::min(),std::numeric_limits<DevicePixelType>::max());
		incrementBufferNumber();
		retriveCurChunk();
	}

	saveChunks(imOut);

	return imOut;
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
		if (!getNextBuffer()->setDims(getCurrentBuffer()->getDims()))
			throw std::runtime_error("Unable to load chunk to the device because the buffer was too small!");

		cudaMeanFilter<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),neighborhood);
		incrementBufferNumber();
		retriveCurChunk();
	}

	saveChunks(meanImage);

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