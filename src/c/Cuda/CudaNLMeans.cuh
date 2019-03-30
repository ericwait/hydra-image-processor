#pragma once
#include "CudaImageContainer.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaUtilities.h"
#include "CudaDeviceInfo.h"
#include "Kernel.cuh"
#include "KernelIterator.cuh"
#include "ImageDimensions.cuh"
#include "ImageChunk.h"
#include "Defines.h"
#include "Vec.h"

#include <cuda_runtime.h>
#include <limits>
#include <omp.h>


// this is an approximate nl means. uses fisher discriminant as a distance, with mean, variance of patches computed by previous cuda filter
template <class PixelTypeIn, class PixelTypeOut>
// params needed: a,h, search window size, comparison nhood
__global__ void cudaNLMeans_mv(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOutMean, CudaImageContainer<PixelTypeOut> imageVariance,
	double h, int searchWindowRadius, int nhoodRadius, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<std::size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		Vec<int> searchVec = Vec<int>::min(Vec<int>(imageIn.getDims()) - 1, Vec<int>(searchWindowRadius));

		// 
		float inputVal = (float)imageIn(threadCoordinate);
		float inputMeanVal = (float)imageOutMean(threadCoordinate);
		float inputVarVal = (float)imageVariance(threadCoordinate);
		// 
		double wMax = 0.;
		double wAccumulator = 0.;
		double outputAccumulator = 0.;
		KernelIterator kIt(threadCoordinate, imageIn.getDims(), searchVec * 2 + 1);
		for (; !kIt.end(); ++kIt)
		{
			Vec<float> kernelPos = kIt.getImageCoordinate();
			if (kernelPos == threadCoordinate)
				continue;

			float kernelMeanVal = (float)imageOutMean(kernelPos);
			float kernelVarVal = (float)imageVariance(kernelPos);
			float kernelVal = (float)imageIn(kernelPos);

			double w = SQR(inputMeanVal - kernelMeanVal) / (inputVarVal + kernelVarVal);
			w= exp(-w / SQR(h));
			if (w > wMax)
				wMax = w;
			wAccumulator += w;
			outputAccumulator += w*kernelVal;
		}
		// add in the value at threadCoordinate, weighted by wMax
		outputAccumulator += wMax * inputVal;
		wAccumulator += wMax;

		// now normalize
		double outVal = outputAccumulator / wAccumulator;

		imageOutMean(threadCoordinate) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
	}
} // cudaNLMeans_mv

template <class PixelTypeIn, class PixelTypeOut>
// params needed: a,h, search window size, comparison nhood
__global__ void cudaNLMeans(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, double h, 
	int searchWindowRadius, int nhoodRadius, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<std::size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		// for loop over whole image, except for searchWindowSize
		Vec<int> itc = Vec<int>(threadCoordinate);

		Vec<int> nhoodVec = Vec<int>::min(Vec<int>(imageIn.getDims())-1, Vec<int>(nhoodRadius));

		Vec<int> searchMin = Vec<int>::max(itc - searchWindowRadius, nhoodVec);
		Vec<int> searchMax = Vec<int>::min(itc + searchWindowRadius+1, Vec<int>(imageIn.getDims()) - nhoodVec);

		// outValAccumulator is unnormalized output value -- we normalize at the end
		double outValAccumulator= 0.0; // 
		// wAccumulator is for the normalizing constant
		double wAccumulator = 1e-7; // 
		double wMax = 0; // for application when x==y==z==0
		for (int z = searchMin.z; z < searchMax.z; z++) {
			for (int y = searchMin.y; y < searchMax.y; y++) {
				for (int x = searchMin.x; x < searchMax.x; x++) {
										
					if ((0 == x) && (0 == y) && (0 == z))
						continue;

					// center the kernel on threadCoordinate
					// iterator over nhoodSize
					KernelIterator kIt(threadCoordinate, imageIn.getDims(), nhoodVec*2+1);
					// image value at the x,y,z neighborhood window center
					float searchVal = (float)imageIn( Vec<float>(x,y,z));
					float wi=0.;
					for (; !kIt.end(); ++kIt)
					{
						Vec<float> imInPos = kIt.getImageCoordinate();
						float inVal = (float)imageIn(imInPos);
						 
						// now get the corresponding pixel value in the search window space
						Vec<float> sCoord = Vec<float>(x,y,z)-nhoodVec+ kIt.getKernelCoordinate();
						float sVal = (float)imageIn(sCoord);

						double diff = SQR(inVal - sVal);
						// weight by gaussian (Ga from paper)
						double ga = 1; // exp(-1 * (Vec<int>(kIt.getKernelCoordinate()) - nhoodVec).lengthSqr() / SQR(a));
						// w is gaussian weighted euclidean distance
						wi += ga*diff;
					}
					double w = exp(-wi / SQR(h));

					if (w > wMax)
						wMax = w;

					outValAccumulator += w*searchVal;
					wAccumulator += w;
				}
			}
		}
		// add in for x==y==z==0
		outValAccumulator += wMax*(float)imageIn(threadCoordinate);
		wAccumulator += wMax;


		// now normalize
		double outVal = outValAccumulator / wAccumulator;
				
		imageOut(threadCoordinate) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
	}
} // cudaNLMeans 

// this part is the templated function that gets called by the front end.
// here be cpu
// todo - if we chunk, make sure search window doesn't go off chunk
template <class PixelTypeIn, class PixelTypeOut>
void cNLMeans(ImageView<PixelTypeIn> imageIn, ImageView<PixelTypeOut> imageOut, double h, int searchWindowRadius, int nhoodRadius, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 3;

	CudaDevices cudaDevs(cudaNLMeans_mv<PixelTypeIn, PixelTypeOut>, device);

	std::size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, Vec<std::size_t>(2*nhoodRadius+1));

	Vec<std::size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	Vec<int> kernelDims = Vec<int>(1+2*nhoodRadius);
	float* kernelMem = new float[kernelDims.product()];
	for (int i = 0; i < kernelDims.product(); i++)
		kernelMem[i] = 1.0;
	ImageView<float> kernel(kernelMem, kernelDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		Kernel constKernelMem(kernel, CUR_DEVICE);

		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());
			
			cudaMeanAndVariance << <chunks[i].blocks, chunks[i].threads >> > (*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), *(deviceImages.getThirdBuffer()), constKernelMem, MIN_VAL, MAX_VAL);

			cudaNLMeans_mv <<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), *(deviceImages.getThirdBuffer()),
				h, searchWindowRadius, nhoodRadius, MIN_VAL, MAX_VAL);
			
			deviceImages.incrementBuffer();
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}
	}
}
