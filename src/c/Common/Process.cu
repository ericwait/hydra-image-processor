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

void set(Vec<unsigned int> imageDims)
{
	if (g_cudaBuffer==NULL)
		g_cudaBuffer = new CudaProcessBuffer<HostPixelType>(imageDims);
}

void set2(Vec<unsigned int> imageDims)
{
	if (g_cudaBuffer2==NULL)
		g_cudaBuffer2 = new CudaProcessBuffer<HostPixelType>(imageDims);
}

void calculateChunks(Vec<unsigned int> imageDims, Vec<unsigned int>& numChunks, Vec<unsigned int>& sizes, int deviceNum=0,
					 bool secondBufferNeeded=false)
{
	HANDLE_ERROR(cudaSetDevice(deviceNum));

	cudaDeviceProp deviceProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,deviceNum));

	unsigned int numVoxels = (unsigned int)(((double)deviceProp.totalGlobalMem/sizeof(HostPixelType))*0.8/NUM_BUFFERS/(1+(int)secondBufferNeeded));

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
			Vec<unsigned int> dims;
			unsigned int dim = (unsigned int)pow((float)numVoxels,1/3.0f);
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

			numChunks.x = (unsigned int)ceil((float)imageDims.x/sizes.x);
			numChunks.y = (unsigned int)ceil((float)imageDims.y/sizes.y);
			numChunks.z = (unsigned int)ceil((float)imageDims.z/sizes.z);
		}
	}
}

void addConstant(const ImageContainer* image,  ImageContainer* imageOut, double additive, int deviceNum)
{
	Vec<unsigned int> numChunks;
	Vec<unsigned int> sizes;
	calculateChunks(image->getDims(),numChunks,sizes);
	if (numChunks.x>1)
		clear2();

	set(sizes);

	Vec<unsigned int> curChunk(0,0,0);
	for (curChunk.z=0; curChunk.z<numChunks.z; ++curChunk.z)
	{
		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
		{
			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
			{
				Vec<unsigned int> curStarts(curChunk.x*sizes.x,curChunk.y*sizes.y,curChunk.z*sizes.z);
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

void contrastEnhancement(const ImageContainer* image, ImageContainer* imageOut, Vec<float> sigmas, Vec<unsigned int> medianNeighborhood)
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

void maxFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<unsigned int> neighborhood, double* kernel)
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

void meanFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<unsigned int> neighborhood )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->meanFilter(neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void medianFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<unsigned int> neighborhood )
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->medianFilter(neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void minFilter( const ImageContainer* image, ImageContainer* imageOut, Vec<unsigned int> neighborhood,
			   double* kernel)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->minFilter(neighborhood,kernel);

	g_cudaBuffer->retrieveImage(imageOut);
}

void morphClosure( const ImageContainer* image, ImageContainer* imageOut, Vec<unsigned int> neighborhood,
			   double* kernel)
{
	set(image->getDims());
	g_cudaBuffer->loadImage(image);

	g_cudaBuffer->morphClosure(neighborhood,kernel);

	g_cudaBuffer->retrieveImage(imageOut);
}

void morphOpening( const ImageContainer* image, ImageContainer* imageOut, Vec<unsigned int> neighborhood,
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
	Vec<unsigned int> numChunks;
	Vec<unsigned int> chunkDims;
	calculateChunks(image->getDims(),numChunks,chunkDims);
	if (numChunks.x>1)
		clear2();

	Vec<unsigned int>reducedImageDims(Vec<unsigned int>(
		(unsigned int)(image->getDims().x/reductions.x),
		(unsigned int)(image->getDims().y/reductions.y),
		(unsigned int)(image->getDims().z/reductions.z)));

	Vec<unsigned int> reducedChunkDims(
		(unsigned int)(chunkDims.x/reductions.x),
		(unsigned int)(chunkDims.y/reductions.y),
		(unsigned int)(chunkDims.z/reductions.z));

	set(chunkDims);

	HostPixelType* curROIorg = new HostPixelType[chunkDims.product()];
	memset(curROIorg,0,chunkDims.product());

	HostPixelType* curROIprocessed = new HostPixelType[reducedChunkDims.product()];
	memset(curROIprocessed,0,reducedChunkDims.product());

	*imageOut = new ImageContainer(reducedImageDims);
	memset((*imageOut)->getMemoryPointer(),0,reducedImageDims.product());

	Vec<unsigned int> curChunk(0,0,0);
	for (curChunk.z=0; curChunk.z<numChunks.z; ++curChunk.z)
	{
		for (curChunk.y=0; curChunk.y<numChunks.y; ++curChunk.y)
		{
			for (curChunk.x=0; curChunk.x<numChunks.x; ++curChunk.x)
			{
				Vec<unsigned int> curStarts(
					curChunk.x*chunkDims.x,
					curChunk.y*chunkDims.y,
					curChunk.z*chunkDims.z);

				Vec<unsigned int> reducedStarts(
					(unsigned int)(curStarts.x/reductions.x),
					(unsigned int)(curStarts.y/reductions.y),
					(unsigned int)(curStarts.z/reductions.z));

				Vec<unsigned int> curChunkDims(
					min(chunkDims.x,image->getWidth()-curStarts.x),
					min(chunkDims.y,image->getHeight()-curStarts.y),
					min(chunkDims.z,image->getDepth()-curStarts.z));

				Vec<unsigned int> curReducedChunkDims(
					(unsigned int)(curChunkDims.x/reductions.x),
					(unsigned int)(curChunkDims.y/reductions.y),
					(unsigned int)(curChunkDims.z/reductions.z));
				
				const HostPixelType* curROIorg = image->getConstROIData(curStarts,curChunkDims);

				g_cudaBuffer->loadImage(curROIorg,curChunkDims);

				g_cudaBuffer->reduceImage(reductions);

				HostPixelType* curROIprocessed = g_cudaBuffer->retrieveImage();

				(*imageOut)->setROIData(curROIprocessed,curStarts,curReducedChunkDims);

				delete[] curROIprocessed;
				delete[] curROIorg;
			}
		}
	}
}

unsigned int* retrieveHistogram(const ImageContainer* image, int& returnSize)
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