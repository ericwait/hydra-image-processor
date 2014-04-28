#pragma once
#include "Vec.h"
#include <vector>
#include <limits>
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaUtilities.cuh"

#include "CudaGaussianFilter.cuh"
#include "CudaAdd.cuh"
#include "CudaMedianFilter.cuh"

template <class PixelType>
PixelType* contrastEnhancement(const PixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood,
							   PixelType** imageOut=NULL, int device=0)
{
	PixelType* imGauss = gaussianFilter<PixelType>(imageIn,dims,sigmas,NULL,device);

	PixelType* imSub = addImageWith<PixelType>(imageIn,imGauss,dims,-1.0,NULL,device);

	delete[] imGauss;

	return medianFilter(imSub,dims,neighborhood,imageOut,device);

// 	PixelType* imOut = setUpOutIm(dims, imageOut);
// 
// 	PixelType minVal = std::numeric_limits<PixelType>::lowest();
// 	PixelType maxVal = std::numeric_limits<PixelType>::max();
// 
// 	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);
// 
// 	float* hostKernel;
// 
// 	Vec<int> gaussIterations(0,0,0);
// 	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,&hostKernel,gaussIterations);
// 	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
// 		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));
// 
// 	cudaDeviceProp props;
// 	cudaGetDeviceProperties(&props,device);
// 
// 	size_t availMem, total;
// 	cudaMemGetInfo(&availMem,&total);
// 
// 	std::vector<ImageChunk> chunks = calculateBuffers<float>(dims,3,(size_t)(availMem*MAX_MEM_AVAIL),props,sizeconstKernelDims);
// 
// 	Vec<size_t> maxDeviceDims;
// 	setMaxDeviceDims(chunks, maxDeviceDims);
// 
// 	CudaDeviceImages<float> deviceImages(3,maxDeviceDims,device);
// 
// 	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
// 	{
// 		deviceImages.setAllDims(curChunk->getFullChunkSize());
// 
// 		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
// 
// 		runGaussIterations(gaussIterations, curChunk, deviceImages, sizeconstKernelDims,device);
// 
// 		curChunk->sendROI(imageIn,dims,deviceImages.getNextBuffer());
// 
// 		cudaAddTwoImagesWithFactor<float><<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getNextBuffer()),*(deviceImages.getCurBuffer()),
// 			*(deviceImages.getThirdBuffer()),-1.0,(float)minVal,(float)maxVal);
// 		DEBUG_KERNEL_CHECK();
// 
// 		deviceImages.setNthBuffCurent(3);
// 
// 		runMedianFilter<float>(props, curChunk, neighborhood, deviceImages);
// 
// 		curChunk->retriveROI<float,PixelType>(imOut,dims,deviceImages.getCurBuffer());
// 	}

	//return imOut;
}

//double* contrastEnhancement(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood,
//							   double** imageOut=NULL, int device=0)
//{
//	double* imOut = setUpOutIm(dims, imageOut);
//
//	double minVal = std::numeric_limits<double>::lowest();
//	double maxVal = std::numeric_limits<double>::max();
//
//	neighborhood = neighborhood.clamp(Vec<size_t>(1,1,1),dims);
//
//	float* hostKernel;
//
//	Vec<int> gaussIterations(0,0,0);
//	Vec<size_t> sizeconstKernelDims = createGaussianKernel(sigmas,&hostKernel,gaussIterations);
//	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*
//		(sizeconstKernelDims.x+sizeconstKernelDims.y+sizeconstKernelDims.z)));
//
//	cudaDeviceProp props;
//	cudaGetDeviceProperties(&props,device);
//
//	size_t availMem, total;
//	cudaMemGetInfo(&availMem,&total);
//
//	std::vector<ImageChunk> chunks = calculateBuffers<double>(dims,3,(size_t)(availMem*MAX_MEM_AVAIL),props,sizeconstKernelDims);
//
//	Vec<size_t> maxDeviceDims;
//	setMaxDeviceDims(chunks, maxDeviceDims);
//
//	CudaDeviceImages<double> deviceImages(3,maxDeviceDims,device);
//
//	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
//	{
//		deviceImages.setAllDims(curChunk->getFullChunkSize());
//
//		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
//
//		runGaussIterations(gaussIterations, curChunk, deviceImages, sizeconstKernelDims,device);
//
//		curChunk->sendROI(imageIn,dims,deviceImages.getNextBuffer());
//
//		cudaAddTwoImagesWithFactor<double><<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getNextBuffer()),*(deviceImages.getCurBuffer()),
//			*(deviceImages.getThirdBuffer()),-1.0,minVal,maxVal);
//		DEBUG_KERNEL_CHECK();
//
//		deviceImages.setNthBuffCurent(3);
//
//		runMedianFilter(props, curChunk, neighborhood, deviceImages);
//
//		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
//	}
//
//	return imOut;
//}