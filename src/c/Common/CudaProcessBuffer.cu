#include "CudaUtilities.cuh"
#include "Vec.h"
#include "CudaProcessBuffer.cuh"
#include "CudaDeviceImages.cuh"
#include "CHelpers.h"
#include "CudaAdd.cuh"
#include "CudaMedianFilter.cuh"
#include "CudaGetMinMax.cuh"
#include "CudaGetROI.cuh"
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

DevicePixelType* CudaProcessBuffer::otsuThresholdFilter(const DevicePixelType* imageIn, Vec<size_t> dims, double alpha/*=1.0*/,
														DevicePixelType** imageOut/*=NULL*/)
{
	double thresh = 0.0;// = otsuThresholdValue(imageIn,dims);
	thresh *= alpha;

	return thresholdFilter(imageIn,dims,(DevicePixelType)thresh,imageOut);
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

void CudaProcessBuffer::unmix(const DevicePixelType* image, Vec<size_t> neighborhood)
{
	//neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);
	throw std::logic_error("The method or operation is not implemented.");
}
