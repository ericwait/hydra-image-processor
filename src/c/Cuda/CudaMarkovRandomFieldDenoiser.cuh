#pragma once
#include "Vec.h"
#include "Defines.h"
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaMultAddFilter.cuh"
#include "CudaVariance.cuh"
#include "CudaAdd.cuh"

template <class PixelType>
__global__ void cudaMarkovGradDescent(CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, PixelType delta)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	DeviceVec<size_t> moves[7] =
	{
		DeviceVec<size_t>(1,0,0),
		DeviceVec<size_t>(0,1,0),
		DeviceVec<size_t>(0,0,1),
		DeviceVec<size_t>(1,1,0),
		DeviceVec<size_t>(1,0,1),
		DeviceVec<size_t>(0,1,1),
		DeviceVec<size_t>(1,1,1)
	};

	if (coordinate<imageOut.getDeviceDims())
	{
		double maxDif = 0.0;
		double centerVal = imageIn[coordinate];
		double curVal = centerVal;

		if (coordinate>DeviceVec<size_t>(0,0,0) && coordinate<imageIn.getDeviceDims()-DeviceVec<size_t>(1,1,1))// left, top, and front planes avail
		{
			for (int i=0; i<7; ++i)
			{
				double curDif = SQR(imageIn[coordinate-moves[i]] - imageIn[coordinate+moves[i]]);
				if (maxDif < curDif)
					if (true)
					{
						maxDif = curDif;

						if (SQR(centerVal-imageIn[coordinate-moves[i]]) > SQR(centerVal-imageIn[coordinate+moves[i]]))
						{
							curVal = centerVal - SIGN(centerVal-imageIn[coordinate-moves[i]]) * delta;
						}
						else
						{
							curVal = centerVal - SIGN(centerVal-imageIn[coordinate+moves[i]]) * delta;
						}
					}
			}
		}
		else
		{
			curVal = centerVal;//TODO change this if it looks like it is working!!!!
		}

		imageOut[coordinate] = curVal;

//		double sgn = 0.0;
// 		if (coordinate.x>0)
// 			sgn -= SIGN(centerVal-imageIn[coordinate-DeviceVec<int>(1,0,0)]);
// 
// 		if (coordinate.x<imageIn.getDeviceDims().x-1)
// 			sgn += SIGN(imageIn[coordinate+DeviceVec<int>(1,0,0)]-centerVal);
// 
// 		if (coordinate.y>0)
// 			sgn -= SIGN(centerVal - imageIn[coordinate-DeviceVec<int>(0,1,0)]);
// 
// 		if (coordinate.y<imageIn.getDeviceDims().y-1)
// 			sgn += SIGN(imageIn[coordinate+DeviceVec<int>(0,1,0)] - centerVal);
// 
// 		if (coordinate.z>0)
// 			sgn -= SIGN(centerVal - imageIn[coordinate-DeviceVec<int>(0,0,1)]);
// 
// 		if (coordinate.z<imageIn.getDeviceDims().z-1)
// 			sgn += SIGN(imageIn[coordinate+DeviceVec<int>(0,0,1)] - centerVal);
// 
// 		imageOut[coordinate] = centerVal + SIGN(sgn)*delta;
	}
}

float* cMarkovRandomFieldDenoiser(const float* imageIn, Vec<size_t> dims, int maxIterations, float** imageOut=NULL, int device=0)
{
// MATLAB code to generate kernel
// 	a(:,:,1) = [0 0 0;0 0 0;0 0 0]
// 	a(:,:,2) = [0 0 0;1 -2 1; 0 0 0]
// 	a(:,:,3) = [0 0 0;0 0 0;0 0 0]
// 	b = convn(a,permute(a,[2 1 3]),'same')
//  c = convn(b,permute(a,[3 1 2]),'same')
//  normalizer = sum((c(:).^2))

    cudaSetDevice(device);

	const size_t SINGLE_KERN_DIM = 3;
	const Vec<size_t> KERN_DIMS = Vec<size_t>(SINGLE_KERN_DIM,SINGLE_KERN_DIM,SINGLE_KERN_DIM);
	float hostKernel[SINGLE_KERN_DIM][SINGLE_KERN_DIM][SINGLE_KERN_DIM] = 
		{{{ 1.0f, -2.0f,  1.0f},
		  {-2.0f,  4.0f, -2.0f},
		  { 1.0f, -2.0f,  1.0f}},
		 {{-2.0f,  4.0f, -2.0f},
		  { 4.0f, -8.0f,  4.0f},
		  {-2.0f,  4.0f, -2.0f}},
		 {{ 1.0f, -2.0f,  1.0f},
		  {-2.0f,  4.0f, -2.0f},
		  { 1.0f, -2.0f,  1.0f}}};

	const double NORMALIZER = 216.0;

	float* imOut = setUpOutIm(dims, imageOut);
	float minVal = std::numeric_limits<float>::lowest();
	float maxVal = std::numeric_limits<float>::max();

	float* deviceSum;
	float* hostSum;

	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, hostKernel, sizeof(float)*KERN_DIMS.product()));

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);

	size_t availMem, total;
	cudaMemGetInfo(&availMem,&total);

	std::vector<ImageChunk> chunks = calculateBuffers<float>(dims,2,(size_t)(availMem*MAX_MEM_AVAIL),props,Vec<size_t>(3,3,3));

	Vec<size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	int sumThreads = props.maxThreadsPerBlock;
	int sumMaxBlocks = (int)ceil((double)maxDeviceDims.product()/(sumThreads*2));
	hostSum = new float[sumMaxBlocks];

	HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(float)*sumMaxBlocks));

	CudaDeviceImages<float> deviceImages(3,maxDeviceDims,device);

	for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
	{
		curChunk->sendROI(imageIn,dims,deviceImages.getCurBuffer());
		deviceImages.setAllDims(curChunk->getFullChunkSize());

		cudaMultAddFilter<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),KERN_DIMS,0,false);
		DEBUG_KERNEL_CHECK();
		deviceImages.incrementBuffer();

		curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());
	}

	double noiseEstimator = (cVariance(imOut,dims,device,(float*)NULL) / NORMALIZER) * dims.product();
	double curNoiseEst = 0.0;

	memcpy(imOut,imageIn,sizeof(float)*dims.product());

	int numIter = 0;
	while (curNoiseEst <= noiseEstimator && numIter<maxIterations)
	{
		curNoiseEst = 0.0;

		for (std::vector<ImageChunk>::iterator curChunk=chunks.begin(); curChunk!=chunks.end(); ++curChunk)
		{
			curChunk->sendROI(imOut,dims,deviceImages.getCurBuffer());
			deviceImages.setAllDims(curChunk->getFullChunkSize());

 			cudaMarkovGradDescent<<<curChunk->blocks,curChunk->threads,sizeof(Vec<size_t>)*7>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),1.0f);
 			DEBUG_KERNEL_CHECK();
 			deviceImages.incrementBuffer();

			curChunk->retriveROI(imOut,dims,deviceImages.getCurBuffer());

			curChunk->sendROI(imageIn,dims,deviceImages.getNextBuffer());
			cudaAddTwoImagesWithFactor<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),
				*(deviceImages.getThirdBuffer()),-1.0,minVal,maxVal);
			DEBUG_KERNEL_CHECK();
			deviceImages.setNthBuffCurent(3);

			cudaPow<<<curChunk->blocks,curChunk->threads>>>(*(deviceImages.getCurBuffer()),*(deviceImages.getNextBuffer()),2.0,minVal,maxVal);
			DEBUG_KERNEL_CHECK();
			deviceImages.incrementBuffer();

			int sumBlocks = (int)ceil((double)curChunk->getFullChunkSize().product()/(sumThreads*2));
			size_t sharedMemSize = sizeof(float)*sumThreads;

			cudaSum<<<sumBlocks,sumThreads,sharedMemSize>>>(deviceImages.getCurBuffer()->getConstImagePointer(),deviceSum,
				curChunk->getFullChunkSize().product());
			DEBUG_KERNEL_CHECK();

			HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(float)*sumBlocks,cudaMemcpyDeviceToHost));

			for (int i=0; i<sumBlocks; ++i)
			{
				curNoiseEst += hostSum[i];
			}

			memset(hostSum,0,sizeof(float)*sumMaxBlocks);
		}

 		++numIter;
	}

	HANDLE_ERROR(cudaFree(deviceSum));
	delete[] hostSum;

	return imOut;
}