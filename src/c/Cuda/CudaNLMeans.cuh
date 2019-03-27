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

// this part runs on the actual card
// here be gpu
template <class PixelTypeIn, class PixelTypeOut>
// params needed: a,h, search window size, comparison nhood
__global__ void cudaNLMeans(CudaImageContainer<PixelTypeIn> imageIn, CudaImageContainer<PixelTypeOut> imageOut, double a, double h, 
	int searchWindowRadius, int nhoodRadius, PixelTypeOut minValue, PixelTypeOut maxValue)
{
	Vec<std::size_t> threadCoordinate;
	GetThreadBlockCoordinate(threadCoordinate);

	if (threadCoordinate < imageIn.getDims())
	{
		// for loop over whole image, except for searchWindowSize
		Vec<int> itc = Vec<int>(threadCoordinate);
		// nhoodVec makes singleton dimensions go to 0, so we can handle e.g. 2d images in the for loops below
		Vec<int> nhoodVec = Vec<int>::min(Vec<int>(imageIn.getDims())-1, Vec<int>(nhoodRadius));

		Vec<int> searchMin = Vec<int>::max(itc - searchWindowRadius, nhoodVec);
		Vec<int> searchMax = Vec<int>::min(itc + searchWindowRadius+1, Vec<int>(imageIn.getDims()) - nhoodVec);

		// outValAccumulator is unnormalized output value -- we normalize at the end
		double outValAccumulator= 0.0; // 
		// wAccumulator is for the normalizing constant
		double wAccumulator = 0; // 
		
		for (int z = searchMin.z; z < searchMax.z; z++) {
			for (int y = searchMin.y; y < searchMax.y; y++) {
				for (int x = searchMin.x; x < searchMax.x; x++) {
					Vec<std::size_t> dVector = Vec<std::size_t>(x, y, z) - threadCoordinate;
					// center the kernel on threadCoordinate
					// iterator over nhoodSize
					//KernelIterator kIt(threadCoordinate, imageIn.getDims(), nhoodVec*2+1);
					// image value at the x,y,z neighborhood window center
					float searchVal = (float)imageIn( Vec<float>(x,y,z));
					double wi=0.;
					for (int k = -nhoodRadius; k < nhoodRadius + 1; k++) 
					{
						// if the kernel goes outside the image, skip it
						// k*nhoodVec/nhoodVec zeros k for singleton dimensions
						Vec<float> sCoord = Vec<float>(x, y, z) + Vec<float>(k) * nhoodVec / nhoodVec;
						// use the point on the search kernel with the difference vector to find the 
						// corresponding point in the input kernel
						Vec<float> imInPosRaw = sCoord-dVector;
						Vec<float> imInPos = CLAMP(imInPosRaw, 0, imageIn.getDims());
						if (imInPosRaw != imInPos)
							continue; // kernel got clamped -- bail
						float inVal = (float)imageIn(imInPos);
						float sVal = (float)imageIn(sCoord);

						double diff = SQR(inVal - sVal);
						// weight by gaussian (Ga from paper)
						double ga = exp(-1 * (Vec<int>(k)*nhoodVec / nhoodVec - nhoodVec).lengthSqr() / SQR(a));
						// w is gaussian weighted euclidean distance
						wi += ga*diff;
					}
					double w = exp(-wi / SQR(h));
					outValAccumulator += w*searchVal;
					wAccumulator += w;
				}
			}
		}

		// now normalize
		//wAccumulator = MAX(wAccumulator, 1e-15);
		double outVal = outValAccumulator / wAccumulator;
				
		imageOut(threadCoordinate) = (PixelTypeOut)CLAMP(outVal, minValue, maxValue);
	}
} // cudaNLMeans 

// this part is the templated function that gets called by the front end.
// here be cpu
// todo - if we chunk, make sure search window doesn't go off chunk
template <class PixelTypeIn, class PixelTypeOut>
void cNLMeans(ImageView<PixelTypeIn> imageIn, ImageView<PixelTypeOut> imageOut, double a, double h, int searchWindowRadius, int nhoodRadius, int device = -1)
{
	const PixelTypeOut MIN_VAL = std::numeric_limits<PixelTypeOut>::lowest();
	const PixelTypeOut MAX_VAL = std::numeric_limits<PixelTypeOut>::max();
	const int NUM_BUFF_NEEDED = 2;

	CudaDevices cudaDevs(cudaNLMeans<PixelTypeIn, PixelTypeOut>, device);

	std::size_t maxTypeSize = MAX(sizeof(PixelTypeIn), sizeof(PixelTypeOut));
	std::vector<ImageChunk> chunks = calculateBuffers(imageIn.getDims(), NUM_BUFF_NEEDED, cudaDevs, maxTypeSize, Vec<std::size_t>(2*nhoodRadius+1));

	Vec<std::size_t> maxDeviceDims;
	setMaxDeviceDims(chunks, maxDeviceDims);

	omp_set_num_threads(MIN(chunks.size(), cudaDevs.getNumDevices()));
	#pragma omp parallel default(shared)
	{
		const int CUDA_IDX = omp_get_thread_num();
		const int N_THREADS = omp_get_num_threads();
		const int CUR_DEVICE = cudaDevs.getDeviceIdx(CUDA_IDX);

		CudaDeviceImages<PixelTypeOut> deviceImages(NUM_BUFF_NEEDED, maxDeviceDims, CUR_DEVICE);
		
		for (int i = CUDA_IDX; i < chunks.size(); i += N_THREADS)
		{
			if (!chunks[i].sendROI(imageIn, deviceImages.getCurBuffer()))
				std::runtime_error("Error sending ROI to device!");

			deviceImages.setAllDims(chunks[i].getFullChunkSize());

			
			cudaNLMeans <<<chunks[i].blocks, chunks[i].threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), 
				a, h, searchWindowRadius, nhoodRadius, MIN_VAL, MAX_VAL);
			deviceImages.incrementBuffer();
			
			chunks[i].retriveROI(imageOut, deviceImages.getCurBuffer());
		}
	}
}
