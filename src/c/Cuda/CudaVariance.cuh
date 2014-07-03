#pragma once
#include "Vec.h"
#include "CudaSum.cuh"
#include "CudaAdd.cuh"
#include "CudaPow.cuh"
#include "CudaConvertType.cuh"

template <class PixelType>
std::vector<ImageChunk> getChunks(const PixelType* im, Vec<size_t> dims, double& minVal, double& maxVal, size_t& sharedMemElementSize, int device)
{
	minVal = (double)std::numeric_limits<float>::lowest();
	maxVal = (double)std::numeric_limits<float>::max();

	sharedMemElementSize = sizeof(float);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t memAvail, total;
	cudaMemGetInfo(&memAvail,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<float>(dims,2,(size_t)(memAvail*MAX_MEM_AVAIL),props);
	return chunks;
}

std::vector<ImageChunk> getChunks(const double* im, Vec<size_t> dims, double& minVal, double& maxVal, size_t& sharedMemElementSize, int device)
{
	minVal = std::numeric_limits<double>::lowest();
	maxVal = std::numeric_limits<double>::max();

	sharedMemElementSize = sizeof(double);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t memAvail, total;
	cudaMemGetInfo(&memAvail,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<double>(dims,2,(size_t)(memAvail*MAX_MEM_AVAIL),props);
	return chunks;
}

template <class PixelType>
double lastSum(const PixelType* im, Vec<size_t> dims, size_t numThreads, size_t numBlocks)
{
	size_t sharedMemSize = numThreads * sizeof(double);
	double sumDouble = 0;
	double* deviceSumDouble;
	double* hostSumDouble;
	HANDLE_ERROR(cudaMalloc((void**)&deviceSumDouble,sizeof(double)*numBlocks));
	hostSumDouble = new double[numBlocks];
	cudaSum<<<numBlocks,numThreads,sharedMemSize>>>(im,deviceSumDouble,dims.product());
	DEBUG_KERNEL_CHECK();

	HANDLE_ERROR(cudaMemcpy(hostSumDouble,deviceSumDouble,sizeof(double)*numBlocks,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceSumDouble));
	deviceSumDouble = NULL;

	for (int i=0; i<numBlocks; ++i)
		sumDouble += hostSumDouble[i];

	delete[] hostSumDouble;

	return sumDouble / dims.product();
}

template <class PixelType>
double squareOfDifferences(size_t elementSize, Vec<size_t> maxDeviceDims, int device, std::vector<ImageChunk> &chunks,
						 const PixelType* imageIn, float* imageOut, Vec<size_t> &dims, float minVal, float maxVal)
{
	size_t numBlocks = chunks[0].blocks.x*chunks[0].blocks.y*chunks[0].blocks.z;
	size_t numThreads = chunks[0].threads.x*chunks[0].threads.y*chunks[0].threads.z;

	size_t sumSize_t = 0;
	size_t* deviceSumSize_t;
	size_t* hostSumSize_t;
	HANDLE_ERROR(cudaMalloc((void**)&deviceSumSize_t,sizeof(size_t)*numBlocks));
	hostSumSize_t = new size_t[numBlocks];

	CudaDeviceImages<PixelType> deviceImages(1,maxDeviceDims,device);
	chunks[0].sendROI(imageIn,dims,deviceImages.getCurBuffer());

	size_t sharedMemSize = numThreads * sizeof(size_t);
	cudaSum<<<numBlocks,numThreads,sharedMemSize>>>(deviceImages.getCurBuffer()->getConstImagePointer(),deviceSumSize_t,dims.product());
	DEBUG_KERNEL_CHECK();

	HANDLE_ERROR(cudaMemcpy(hostSumSize_t,deviceSumSize_t,sizeof(size_t)*numBlocks,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceSumSize_t));
	deviceSumSize_t = NULL;

	for (int i=0; i<numBlocks; ++i)
		sumSize_t += hostSumSize_t[i];

	delete[] hostSumSize_t;

	float imMean = (float)sumSize_t / (float)dims.product();

	CudaDeviceImages<float> deviceImagesF1(1,maxDeviceDims,device);
	cudaConvertType<<<numBlocks,numThreads>>>(deviceImages.getCurBuffer()->getConstImagePointer(),
		deviceImagesF1.getCurBuffer()->getDeviceImagePointer(),dims.product(),minVal,maxVal);
	deviceImages.clear();

	CudaDeviceImages<float> deviceImagesF2(1,maxDeviceDims,device);
	cudaAddScaler<<<chunks[0].blocks,chunks[0].threads>>>(*(deviceImagesF1.getCurBuffer()),*(deviceImagesF2.getCurBuffer()),-imMean,minVal,maxVal);
	DEBUG_KERNEL_CHECK();

	if (imageOut!=NULL)
		HANDLE_ERROR(cudaMemcpy(imageOut,deviceImagesF2.getCurBuffer()->getConstImagePointer(),sizeof(float)*dims.product(),cudaMemcpyDeviceToHost));

	cudaPow<<<chunks[0].blocks,chunks[0].threads>>>(*(deviceImagesF2.getCurBuffer()),*(deviceImagesF1.getCurBuffer()),2.0,minVal,maxVal);
	DEBUG_KERNEL_CHECK();

	return lastSum(deviceImagesF1.getCurBuffer()->getConstImagePointer(),maxDeviceDims,numThreads,numBlocks);
}

double squareOfDifferences(size_t elementSize, Vec<size_t> maxDeviceDims, int device, std::vector<ImageChunk> &chunks,
						 const float* imageIn, float* imageOut, Vec<size_t> &dims, float minVal, float maxVal)
{
	size_t numBlocks = chunks[0].blocks.x*chunks[0].blocks.y*chunks[0].blocks.z;
	size_t numThreads = chunks[0].threads.x*chunks[0].threads.y*chunks[0].threads.z;

	double sumVal = 0;
	double* deviceSum;
	double* hostSum;
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*numBlocks));
	hostSum= new double[numBlocks];

	CudaDeviceImages<float> deviceImages(2,maxDeviceDims,device);
	chunks[0].sendROI(imageIn,dims,deviceImages.getCurBuffer());
	deviceImages.setAllDims(maxDeviceDims);

	size_t sharedMemSize = numThreads * sizeof(double);
	cudaSum<<<numBlocks,numThreads,sharedMemSize>>>(deviceImages.getCurBuffer()->getConstImagePointer(),deviceSum,dims.product());
	DEBUG_KERNEL_CHECK();

	HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*numBlocks,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceSum));
	deviceSum= NULL;

	for (int i=0; i<numBlocks; ++i)
		sumVal += hostSum[i];

	delete[] hostSum;

	double imMean = sumVal / dims.product();

	cudaAddScaler<<<chunks[0].blocks,chunks[0].threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),-imMean,minVal,maxVal);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();

	if (imageOut!=NULL)
		HANDLE_ERROR(cudaMemcpy(imageOut,deviceImages.getCurBuffer()->getConstImagePointer(),sizeof(float)*dims.product(),cudaMemcpyDeviceToHost));


	cudaPow<<<chunks[0].blocks,chunks[0].threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),2.0,minVal,maxVal);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();

	return lastSum(deviceImages.getCurBuffer()->getConstImagePointer(),maxDeviceDims,numThreads,numBlocks);
}

double squareOfDifferences(size_t elementSize, Vec<size_t> maxDeviceDims, int device, std::vector<ImageChunk> &chunks,
						   const double* imageIn, double* imageOut, Vec<size_t> &dims, double minVal, double maxVal)
{
	size_t numBlocks = chunks[0].blocks.x*chunks[0].blocks.y*chunks[0].blocks.z;
	size_t numThreads = chunks[0].threads.x*chunks[0].threads.y*chunks[0].threads.z;

	double sumVal = 0;
	double* deviceSum;
	double* hostSum;
	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*numBlocks));
	hostSum= new double[numBlocks];

	CudaDeviceImages<double> deviceImages(2,maxDeviceDims,device);
	chunks[0].sendROI(imageIn,dims,deviceImages.getCurBuffer());
	deviceImages.setAllDims(maxDeviceDims);

	size_t sharedMemSize = numThreads * sizeof(double);
	cudaSum<<<numBlocks,numThreads,sharedMemSize>>>(deviceImages.getCurBuffer()->getConstImagePointer(),deviceSum,dims.product());
	DEBUG_KERNEL_CHECK();

	HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*numBlocks,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(deviceSum));
	deviceSum= NULL;

	for (int i=0; i<numBlocks; ++i)
		sumVal += hostSum[i];

	delete[] hostSum;

	double imMean = sumVal / dims.product();

	cudaAddScaler<<<chunks[0].blocks,chunks[0].threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),-imMean,minVal,maxVal);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();

	if (imageOut!=NULL)
		HANDLE_ERROR(cudaMemcpy(imageOut,deviceImages.getCurBuffer()->getConstImagePointer(),sizeof(double)*dims.product(),cudaMemcpyDeviceToHost));


	cudaPow<<<chunks[0].blocks,chunks[0].threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),2.0,minVal,maxVal);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();

	return lastSum(deviceImages.getCurBuffer()->getConstImagePointer(),maxDeviceDims,numThreads,numBlocks);
}

template <class PixelType, class PixelTypeOut> //do something different for float and double
double cVariance(const PixelType* imageIn, Vec<size_t> dims, int device=0, PixelTypeOut* imageOut=NULL)
{
	double variance = 0.0;

	size_t elementSize;
	double minVal, maxVal;
	std::vector<ImageChunk> chunks = getChunks(imageIn,dims,minVal,maxVal,elementSize,device);

	if (chunks.size()==1)
	{
		Vec<size_t> maxDeviceDims;
		setMaxDeviceDims(chunks, maxDeviceDims);

		variance = squareOfDifferences(elementSize, maxDeviceDims, device, chunks, imageIn, imageOut, dims, minVal, maxVal);
	}
	else
	{
		double imMean = cSumArray<double>(imageIn,dims.product(),device) / (double)dims.product();
		float* imSub = cAddConstant<PixelType,float>(imageIn,dims,-imMean,NULL,device);
		float* imP = cImagePow<float>(imSub,dims,2.0,NULL,device);
		variance = cSumArray<double>(imP,dims.product(),device)/(double)dims.product();

		delete[] imSub;
		delete[] imP;
	}

	return variance;
}
