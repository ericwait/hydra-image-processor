#include "Process.h"
#include "CHelpers.h"
#include "CudaUtilities.cuh"
#include "CudaProcessBuffer.cuh"

CudaProcessBuffer<HostPixelType>* g_cudaBuffer = NULL;
CudaProcessBuffer<HostPixelType>* g_cudaBuffer2 = NULL;

void clear1()
{
	if (g_cudaBuffer!=NULL)
	{
		delete g_cudaBuffer;
		g_cudaBuffer = NULL;
	}
}

void clear2()
{
	if (g_cudaBuffer2!=NULL)
	{
		delete g_cudaBuffer2;
		g_cudaBuffer2 = NULL;
	}
}


void clearAll()
{
	clear1();
	clear2();
}

void set(Vec<size_t> imageDims)
{
	if (g_cudaBuffer==NULL)
		g_cudaBuffer = new CudaProcessBuffer<HostPixelType>(imageDims);
}

void set2(Vec<size_t> imageDims)
{
	if (g_cudaBuffer2==NULL)
		g_cudaBuffer2 = new CudaProcessBuffer<HostPixelType>(imageDims);
}

void calculateChunks(Vec<size_t> imageDims, Vec<size_t>& numChunks, Vec<size_t>& sizes, int deviceNum=0,
					 bool secondBufferNeeded=false)
{
	HANDLE_ERROR(cudaSetDevice(deviceNum));

	cudaDeviceProp deviceProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,deviceNum));

	size_t numVoxels = (size_t)(((double)deviceProp.totalGlobalMem/sizeof(HostPixelType))*0.8/NUM_BUFFERS/(1+(int)secondBufferNeeded));

	if (imageDims.z==1)
	{
		if (imageDims.y==1)
		{
			if (imageDims.x<numVoxels)
			{
				sizes.x = imageDims.x;
				sizes.y = 1;
				sizes.z = 1;

				numChunks.x = 1;
				numChunks.y = 1;
				numChunks.z = 1;
			} 
			else
			{
				sizes.x = numVoxels;
				sizes.y = 1;
				sizes.z = 1;

				numChunks.x = (int)ceil((float)imageDims.x/numVoxels);
				numChunks.y = 1;
				numChunks.z = 1;
			}
		}
		else
		{
			if (imageDims.x*imageDims.y<numVoxels)
			{
				sizes.x = imageDims.x;
				sizes.y = imageDims.y;
				sizes.z = 1;

				numChunks.x = 1;
				numChunks.y = 1;
				numChunks.z = 1;
			} 
			else
			{
				int dim = (int)sqrt((double)numVoxels);

				sizes.x = dim;
				sizes.y = dim;
				sizes.z = 1;

				numChunks.x = (int)ceil((float)imageDims.x/dim);
				numChunks.y = (int)ceil((float)imageDims.y/dim);
				numChunks.z = 1;
			}
		}
	}
	else
	{
		if(imageDims.product()<numVoxels)
		{
			sizes.x = imageDims.x;
			sizes.y = imageDims.y;
			sizes.z = imageDims.z;

			numChunks.x = 1;
			numChunks.y = 1;
			numChunks.z = 1;
		}
		else
		{
			Vec<size_t> dims;
			size_t dim = (size_t)pow((float)numVoxels,1/3.0f);
			if (dim>imageDims.z)
			{
				dim = (int)sqrt((double)numVoxels/imageDims.z);
				dims.z = imageDims.z;
				dims.x = dim;
				dims.y = dim;
			}

			float extra = (float)(numVoxels-dims.x*dims.y*dims.z)/(dims.x*dims.y);

			sizes.x = dims.x + (int)extra;
			sizes.y = dims.y;
			sizes.z = dims.z;

			numChunks.x = (size_t)ceil((float)imageDims.x/sizes.x);
			numChunks.y = (size_t)ceil((float)imageDims.y/sizes.y);
			numChunks.z = (size_t)ceil((float)imageDims.z/sizes.z);
		}
	}
}

void addConstant(const ImageContainer* image,  ImageContainer* imageOut, double additive, int deviceNum)
{
	Vec<size_t> numChunks;
	Vec<size_t> sizes;
	calculateChunks(image->getDims(),numChunks,sizes);
	if (numChunks.x>1)
		clear2();

	set(sizes);

	Vec<size_t> curChunk(0,0,0);
	for (curChunk.z=0; curChunk.z<numChunks.z; ++curChunk.z)
	{
		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
		{
			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
			{
				Vec<size_t> curStarts(curChunk.x*sizes.x,curChunk.y*sizes.y,curChunk.z*sizes.z);
				const HostPixelType* curROIorg = image->getConstROIData(curStarts,sizes);

				g_cudaBuffer->loadImage(curROIorg,sizes);

				g_cudaBuffer->addConstant(additive);

				HostPixelType* curROIprocessed = g_cudaBuffer->retrieveImage();

				imageOut->setROIData(curROIprocessed,curStarts,sizes);

				delete[] curROIprocessed;
				delete[] curROIorg;
			}
		}
	}
}

void addImageWith(const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut, double factor)
{
	set(image1->getDims());
	set2(image2->getDims());
	g_cudaBuffer->loadImage(image1);
	g_cudaBuffer2->loadImage(image2);

	g_cudaBuffer->addImageWith(g_cudaBuffer2,factor);

	g_cudaBuffer->retrieveImage(imageOut);
}

void applyPolyTransformation( const ImageContainer* image, ImageContainer* imageOut, double a, double b, double c, HostPixelType minValue,
							 HostPixelType maxValue )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->applyPolyTransformation(a,b,c,minValue,maxValue);

	g_cudaBuffer->retrieveImage(imageOut);
}

void calculateMinMax(const ImageContainer* image, double& minValue, double& maxValue)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->calculateMinMax(minValue,maxValue);
}

void contrastEnhancement(const ImageContainer* image, ImageContainer* imageOut, Vec<float> sigmas, Vec<size_t> medianNeighborhood)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->contrastEnhancement(sigmas,medianNeighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void gaussianFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<float> sigmas )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->gaussianFilter(sigmas);

	g_cudaBuffer->retrieveImage(imageOut);
}

size_t getGlobalMemoryAvailable()
{
	CudaProcessBuffer<HostPixelType> cudaBuffer(1);
	return g_cudaBuffer->getGlobalMemoryAvailable();
}

void mask(const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut, double threshold)
{
	set(image1->getDims());
	set2(image2->getDims());
	g_cudaBuffer->loadImage(image1);
	g_cudaBuffer2->loadImage(image2);

	g_cudaBuffer->mask(g_cudaBuffer2,(HostPixelType)threshold);

	g_cudaBuffer->retrieveImage(imageOut);
}

void maxFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood, double* kernel)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->maxFilter(neighborhood,kernel);

	g_cudaBuffer->retrieveImage(imageOut);
}

void maximumIntensityProjection( const ImageContainer* image, ImageContainer* imageOut)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->maximumIntensityProjection();

	g_cudaBuffer->retrieveImage(imageOut);
}

void meanFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->meanFilter(neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void medianFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->medianFilter(neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void minFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood,
			   double* kernel)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->minFilter(neighborhood,kernel);

	g_cudaBuffer->retrieveImage(imageOut);
}

void morphClosure( const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood,
			   double* kernel)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->morphClosure(neighborhood,kernel);

	g_cudaBuffer->retrieveImage(imageOut);
}

void morphOpening( const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood,
				  double* kernel)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->morphOpening(neighborhood,kernel);

	g_cudaBuffer->retrieveImage(imageOut);
}

void multiplyImage( const ImageContainer* image, ImageContainer* imageOut, double factor )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->multiplyImage(factor);

	g_cudaBuffer->retrieveImage(imageOut);
}

void multiplyImageWith( const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut )
{
	set(image1->getDims());
	set2(image2->getDims());
	g_cudaBuffer->loadImage(image1);
	g_cudaBuffer2->loadImage(image2);

	g_cudaBuffer->multiplyImageWith(g_cudaBuffer2);

	g_cudaBuffer->retrieveImage(imageOut);
}

double normalizedCovariance(const ImageContainer* image1, const ImageContainer* image2)
{
	set(image1->getDims());
	set2(image2->getDims());
	g_cudaBuffer->loadImage(image1);
	g_cudaBuffer2->loadImage(image2);

	return g_cudaBuffer->normalizedCovariance(g_cudaBuffer2);
}

void otsuThresholdFilter(const ImageContainer* image, ImageContainer* imageOut, double alpha)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->otsuThresholdFilter((float)alpha);

	g_cudaBuffer->retrieveImage(imageOut);
}

HostPixelType otsuThesholdValue(const ImageContainer* image)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	return g_cudaBuffer->otsuThresholdValue();
}

void imagePow( const ImageContainer* image, ImageContainer* imageOut, int p )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->imagePow(p);

	g_cudaBuffer->retrieveImage(imageOut);
}

double sumArray(const ImageContainer* image)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	double sum;
	g_cudaBuffer->sumArray(sum);
	return sum;
}

void reduceImage( const ImageContainer* image, ImageContainer** imageOut, Vec<double> reductions)
{
	Vec<size_t> numChunks;
	Vec<size_t> chunkDims;
	calculateChunks(image->getDims(),numChunks,chunkDims);
	if (numChunks.x>1)
		clear2();

	Vec<size_t>reducedImageDims(Vec<size_t>(
		(size_t)(image->getDims().x/reductions.x),
		(size_t)(image->getDims().y/reductions.y),
		(size_t)(image->getDims().z/reductions.z)));

	Vec<size_t> reducedChunkDims(
		(size_t)(chunkDims.x/reductions.x),
		(size_t)(chunkDims.y/reductions.y),
		(size_t)(chunkDims.z/reductions.z));

	set(chunkDims);

	HostPixelType* curROIorg = new HostPixelType[chunkDims.product()];
	memset(curROIorg,0,chunkDims.product());

	HostPixelType* curROIprocessed = new HostPixelType[reducedChunkDims.product()];
	memset(curROIprocessed,0,reducedChunkDims.product());

	*imageOut = new ImageContainer(reducedImageDims);
	memset((*imageOut)->getMemoryPointer(),0,reducedImageDims.product());

	Vec<size_t> curChunk(0,0,0);
	for (curChunk.z=0; curChunk.z<numChunks.z; ++curChunk.z)
	{
		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
		{
			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
			{
				Vec<size_t> curStarts(
					curChunk.x*chunkDims.x,
					curChunk.y*chunkDims.y,
					curChunk.z*chunkDims.z);

				Vec<size_t> reducedStarts(
					(size_t)(curStarts.x/reductions.x),
					(size_t)(curStarts.y/reductions.y),
					(size_t)(curStarts.z/reductions.z));

				Vec<size_t> curChunkDims(
					min(chunkDims.x,image->getWidth()-curStarts.x),
					min(chunkDims.y,image->getHeight()-curStarts.y),
					min(chunkDims.z,image->getDepth()-curStarts.z));

				Vec<size_t> curReducedChunkDims(
					(size_t)(curChunkDims.x/reductions.x),
					(size_t)(curChunkDims.y/reductions.y),
					(size_t)(curChunkDims.z/reductions.z));
				
				const HostPixelType* curROIorg = image->getConstROIData(curStarts,curChunkDims);

				g_cudaBuffer->loadImage(curROIorg,curChunkDims);

				g_cudaBuffer->reduceImage(reductions);

				HostPixelType* curROIprocessed = g_cudaBuffer->retrieveImage();

				(*imageOut)->setROIData(curROIprocessed,reducedStarts,curReducedChunkDims);

				delete[] curROIprocessed;
				delete[] curROIorg;
			}
		}
	}
}

size_t* retrieveHistogram(const ImageContainer* image, int& returnSize)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	return g_cudaBuffer->retrieveHistogram(returnSize);
}

double* retrieveNormalizedHistogram(const ImageContainer* image, int& returnSize)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	return g_cudaBuffer->retrieveNormalizedHistogram(returnSize);
}

void thresholdFilter( const ImageContainer* image, ImageContainer* imageOut, double threshold )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->thresholdFilter(threshold);

	g_cudaBuffer->retrieveImage(imageOut);
}