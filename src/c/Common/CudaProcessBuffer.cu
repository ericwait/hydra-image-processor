#include "CudaUtilities.cuh"
#include "Vec.h"
#include "CudaProcessBuffer.cuh"
#include "CudaDeviceImages.cuh"
#include "CHelpers.h"
#include "CudaAdd.cuh"
#include "CudaMedianFilter.cuh"
#include "CudaGetMinMax.cuh"
#include "CudaGetROI.cuh"
#include "CudaHistogramCreate.cuh"
#include "CudaMask.cuh"
#include "CudaMaxFilter.cuh"
#include "CudaIntensityProjection.cuh"
#include "CudaMeanFilter.cuh"
#include "CudaImageReduction.cuh"
#include "CudaMinFilter.cuh"
#include "CudaMultAddFilter.cuh"
#include "CudaMultiplyImage.cuh"
#include "CudaPolyTransferFunc.cuh"
#include "CudaPow.cuh"
#include "CudaSum.cuh"
#include "CudaThreshold.cuh"
#include "CudaUnmixing.cuh"
#include "CudaPow.cuh"
#include "CudaGaussianFilter.cuh"

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



//////////////////////////////////////////////////////////////////////////
//Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

DevicePixelType* CudaProcessBuffer::applyPolyTransformation(const DevicePixelType* imageIn, Vec<size_t> dims, double a, double b, double c,
												DevicePixelType minValue, DevicePixelType maxValue, DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);
	
	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaPolyTransferFunc<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
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

	float* hostKernel;

	Vec<int> gaussIterations(0,0,0);
	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,&hostKernel,gaussIterations);
	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,3,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,
		sizeconstKernelDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(3,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		deviceImages.setAllDims(curChunk->getFullChunkSize());

		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());

		runGaussIterations(gaussIterations, curChunk, deviceImages, sizeconstKernelDims,device);

		curChunk->sendROI(imageIn,dims,deviceImages.getNextBuffer());

 		cudaAddTwoImagesWithFactor<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getNextBuffer()),*(deviceImages.getCurBuffer()),
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

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,1,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);
	setMaxDeviceDims(chunks, maxDeviceDims);
	CudaDeviceImages<DevicePixelType> deviceImages(1,maxDeviceDims,device);

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

void CudaProcessBuffer::getMinMax(const DevicePixelType* imageIn, size_t n, DevicePixelType& minVal, DevicePixelType& maxVal)
{
	minVal = std::numeric_limits<DevicePixelType>::lowest();
	maxVal= std::numeric_limits<DevicePixelType>::max();
	double* deviceMin;
	double* deviceMax;
	double* hostMin;
	double* hostMax;
	DevicePixelType* deviceImage;

	unsigned int blocks = deviceProp.multiProcessorCount;
	unsigned int threads = deviceProp.maxThreadsPerBlock;

	Vec<size_t> maxDeviceDims(1,1,1);

	maxDeviceDims.x = (n < (double)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL)/sizeof(DevicePixelType)) ? (n) :
		((size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL/sizeof(DevicePixelType)));

	checkFreeMemory(sizeof(DevicePixelType)*maxDeviceDims.x+sizeof(double)*blocks,device,true);
	HANDLE_ERROR(cudaMalloc((void**)&deviceImage,sizeof(DevicePixelType)*maxDeviceDims.x));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMin,sizeof(double)*blocks));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMax,sizeof(double)*blocks));
	hostMin = new double[blocks];
	hostMax = new double[blocks];

	for (int i=0; i<ceil((double)n/maxDeviceDims.x); ++i)
	{
		const DevicePixelType* imStart = imageIn + i*maxDeviceDims.x;
		size_t numValues = ((i+1)*maxDeviceDims.x < n) ? (maxDeviceDims.x) : (n-i*maxDeviceDims.x);

		HANDLE_ERROR(cudaMemcpy(deviceImage,imStart,sizeof(DevicePixelType)*numValues,cudaMemcpyHostToDevice));

		cudaGetMinMax<<<blocks,threads,sizeof(double)*threads*2>>>(deviceImage,deviceMin,deviceMax,numValues);
		DEBUG_KERNEL_CHECK();

		HANDLE_ERROR(cudaMemcpy(hostMin,deviceMin,sizeof(double)*blocks,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(hostMax,deviceMax,sizeof(double)*blocks,cudaMemcpyDeviceToHost));

		for (unsigned int i=0; i<blocks; ++i)
		{
			if (minVal> hostMin[i])
				minVal = (DevicePixelType)(hostMin[i]);

			if (maxVal< hostMax[i])
				maxVal = (DevicePixelType)(hostMax[i]);
		}
	}

	HANDLE_ERROR(cudaFree(deviceMin));
	HANDLE_ERROR(cudaFree(deviceMax));
	HANDLE_ERROR(cudaFree(deviceImage));

	delete[] hostMin;
	delete[] hostMax;
}



DevicePixelType* CudaProcessBuffer::maxFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/,
						   DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	if (kernel==NULL)
	{
		kernelDims = kernelDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernelDims.product()];
		for (int i=0; i<kernelDims.product(); ++i)
			ones[i] = 1.0f;

		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(float)*kernelDims.product()));
		delete[] ones;
	} 
	else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*kernelDims.product()));
	}

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,kernelDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMaxFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernelDims,
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

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,neighborhood);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(2,maxDeviceDims,device);

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

DevicePixelType* CudaProcessBuffer::minFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/,
											  DevicePixelType** imageOut/*=NULL*/)
{
	DevicePixelType* imOut = setUpOutIm(dims, imageOut);

	DevicePixelType minVal = std::numeric_limits<DevicePixelType>::lowest();
	DevicePixelType maxVal = std::numeric_limits<DevicePixelType>::max();

	if (kernel==NULL)
	{
		kernelDims = kernelDims.clamp(Vec<size_t>(1,1,1),dims);
		float* ones = new float[kernelDims.product()];
		for (int i=0; i<kernelDims.product(); ++i)
			ones[i] = 1.0f;

		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, ones, sizeof(float)*kernelDims.product()));
		delete[] ones;
	} 
	else
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*kernelDims.product()));
	}

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp,kernelDims);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMinFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),kernelDims,
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

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaMultiplyImageScaler<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
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

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,3,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(3,maxDeviceDims,device);

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

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,1,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);
	setMaxDeviceDims(chunks, maxDeviceDims);
	CudaDeviceImages<DevicePixelType> deviceImages(1,maxDeviceDims,device);

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

	std::vector<ImageChunk> orgChunks = calculateBuffers<DevicePixelType>(dims,1,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL*(1-ratio)),deviceProp,reductions);
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

	CudaImageContainerClean<DevicePixelType>* deviceImageIn = new CudaImageContainerClean<DevicePixelType>(orgChunks[0].getFullChunkSize(),device);
	CudaImageContainerClean<DevicePixelType>* deviceImageOut = new CudaImageContainerClean<DevicePixelType>(reducedChunks[0].getFullChunkSize(),device);

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

	std::vector<ImageChunk> chunks = calculateBuffers<DevicePixelType>(dims,2,(size_t)(deviceProp.totalGlobalMem*MAX_MEM_AVAIL),deviceProp);

	setMaxDeviceDims(chunks, maxDeviceDims);

	CudaDeviceImages<DevicePixelType> deviceImages(2,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setNextDims(curChunk->getFullChunkSize());

		cudaThreshold<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
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
