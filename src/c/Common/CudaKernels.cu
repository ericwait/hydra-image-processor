#include "CudaKernels.cuh"

__global__ void cudaMeanFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		double val = 0;
		double kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					val += imageIn[curCoordIm];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinate] = val/kernelVolume;
	}
}

__global__ void cudaMultiplyImage( CudaImageContainer imageIn, CudaImageContainer imageOut, double factor, DevicePixelType minValue, DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		imageOut[coordinate] = min((double)maxValue,max((double)minValue, factor*imageIn[coordinate]));
	}
}

__global__ void cudaAddTwoImagesWithFactor( CudaImageContainer imageIn1, CudaImageContainer imageIn2, CudaImageContainer imageOut, double factor, DevicePixelType minValue, DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		double subtractor = factor*(double)imageIn2[coordinate];
		DevicePixelType outValue = (double)imageIn1[coordinate] + subtractor;

		imageOut[coordinate] = min(maxValue,max(minValue,outValue));
	}
}

__global__ void cudaMultiplyTwoImages( CudaImageContainer imageIn1, CudaImageContainer imageIn2, CudaImageContainer imageOut )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		DevicePixelType val1 = imageIn1[coordinate];
		DevicePixelType val2 = imageIn2[coordinate];
		imageOut[coordinate] = imageIn1[coordinate] * imageIn2[coordinate];
	}
}

__global__ void cudaAddFactor( CudaImageContainer imageIn1, CudaImageContainer imageOut, double factor, DevicePixelType minValue,
							  DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		double outValue = imageIn1[coordinate] + factor;
		imageOut[coordinate] = min((double)maxValue,max((double)minValue,outValue));
	}
}

__device__ DevicePixelType* SubDivide(DevicePixelType* pB, DevicePixelType* pE)
{
	DevicePixelType* pPivot = --pE;
	const DevicePixelType pivot = *pPivot;

	while (pB < pE)
	{
		if (*pB > pivot)
		{
			--pE;
			DevicePixelType temp = *pB;
			*pB = *pE;
			*pE = temp;
		} else
			++pB;
	}

	DevicePixelType temp = *pPivot;
	*pPivot = *pE;
	*pE = temp;

	return pE;
}

__device__ void SelectElement(DevicePixelType* pB, DevicePixelType* pE, size_t k)
{
	while (true)
	{
		DevicePixelType* pPivot = SubDivide(pB, pE);
		size_t n = pPivot - pB;

		if (n == k)
			break;

		if (n > k)
			pE = pPivot;
		else
		{
			pB = pPivot + 1;
			k -= (n + 1);
		}
	}
}

__device__ DevicePixelType cudaFindMedian(DevicePixelType* vals, int numVals)
{
	SelectElement(vals,vals+numVals, numVals/2);
	return vals[numVals/2];
}

__global__ void cudaMedianFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims )
{
	extern __shared__ DevicePixelType vals[];
	DeviceVec<size_t> kernelDims = hostKernelDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  kernelDims.product();

	if (coordinate<imageIn.getDims())
	{
		int kernelVolume = 0;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					vals[kernelVolume+offset] = imageIn[curCoordIm];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinate] = cudaFindMedian(vals+offset,kernelVolume);
		__syncthreads();
	}
}

__global__ void cudaMultAddFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims, int kernelOffset/*=0*/ )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		double val = 0;
		double kernFactor = 0;

		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					size_t kernIdx = kernelDims.linearAddressAt(curCoordKrn)+kernelOffset;
					kernFactor += cudaConstKernel[kernIdx];
					val += imageIn[curCoordIm] * cudaConstKernel[kernIdx];
				}
			}
		}

		imageOut[coordinate] = val/kernFactor;
	}
}

__global__ void cudaMinFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		DevicePixelType minVal = imageIn[coordinate];
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					if(cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]>0)
					{
						minVal = (DevicePixelType)min((float)minVal, imageIn[curCoordIm]*
							cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]);
					}
				}
			}
		}

		imageOut[coordinate] = minVal;
	}
}

__global__ void cudaMaxFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		DevicePixelType maxVal = imageIn[coordinate];
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = (kernelDims.x+1)/2;
		kernelMidIdx.y = (kernelDims.y+1)/2;
		kernelMidIdx.z = (kernelDims.z+1)/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					if(cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]>0)
					{
						maxVal = (DevicePixelType)max((float)maxVal, imageIn[curCoordIm]*
							cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]);
					}
				}
			}
		}

		imageOut[coordinate] = maxVal;
	}
}

__global__ void cudaHistogramCreate( CudaImageContainer imageIn, size_t* histogram )
{
	//This code is modified from that of Sanders - Cuda by Example
	__shared__ size_t tempHisto[NUM_BINS];
	tempHisto[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < imageIn.getDims().product())
	{
		atomicAdd(&(tempHisto[imageIn[i]]), 1);
		i += stride;
	}

	__syncthreads();
	atomicAdd(&(histogram[threadIdx.x]), tempHisto[threadIdx.x]);
}

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, Vec<size_t> imageDims)
{
	int x = blockIdx.x;
	normHistogram[x] = (double)(histogram[x]) / (imageDims.x*imageDims.y*imageDims.z);
}

__global__ void cudaThresholdImage( CudaImageContainer imageIn, CudaImageContainer imageOut, DevicePixelType threshold,
								   DevicePixelType minValue, DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		if (imageIn[coordinate]>=threshold)
			imageOut[coordinate] = maxValue;
		else
			imageOut[coordinate] = minValue;
	}
}

__global__ void cudaFindMinMax( CudaImageContainer arrayIn, double* minArrayOut, double* maxArrayOut, size_t n )
{
	extern __shared__ double maxData[];
	extern __shared__ double minData[];

	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockDim.x*2 + tid;
	size_t gridSize = blockDim.x*2*gridDim.x;

	while (i<n)
	{
		maxData[tid] = arrayIn[i];
		minData[tid] = arrayIn[i];

		if (i+blockDim.x<n)
		{
			if(maxData[tid]<arrayIn[i+blockDim.x])
				maxData[tid] = arrayIn[i+blockDim.x];

			if(minData[tid]>arrayIn[i+blockDim.x])
				minData[tid] = arrayIn[i+blockDim.x];
		}

		i += gridSize;
	}
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
		{
			if(maxData[tid]<maxData[tid + 1024])
				maxData[tid] = maxData[tid + 1024];

			if(minData[tid]>minData[tid + 1024])
				minData[tid] = minData[tid + 1024];
		}
		__syncthreads();
	}
	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
		{
			if(maxData[tid]<maxData[tid + 512])
				maxData[tid] = maxData[tid + 512];

			if(minData[tid]>minData[tid + 512])
				minData[tid] = minData[tid + 512];
		}
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) 
		{
			if(maxData[tid]<maxData[tid + 256])
				maxData[tid] = maxData[tid + 256];

			if(minData[tid]>minData[tid + 256])
				minData[tid] = minData[tid + 256];
		}
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
		{
			if(maxData[tid]<maxData[tid + 128])
				maxData[tid] = maxData[tid + 128];

			if(minData[tid]>minData[tid + 128])
				minData[tid] = minData[tid + 128];
		}
		__syncthreads(); 
	}
	if (blockDim.x >= 128) 
	{
		if (tid < 64)
		{
			if(maxData[tid]<maxData[tid + 64])
				maxData[tid] = maxData[tid + 64];

			if(minData[tid]>minData[tid + 64])
				minData[tid] = minData[tid + 64];
		}
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			{
				if(maxData[tid]<maxData[tid + 64])
					maxData[tid] = maxData[tid + 64];

				if(minData[tid]>minData[tid + 64])
					minData[tid] = minData[tid + 64];
			}
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			if(maxData[tid]<maxData[tid + 16])
				maxData[tid] = maxData[tid + 16];

			if(minData[tid]>minData[tid + 16])
				minData[tid] = minData[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			if(maxData[tid]<maxData[tid + 8])
				maxData[tid] = maxData[tid + 8];

			if(minData[tid]>minData[tid + 8])
				minData[tid] = minData[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			if(maxData[tid]<maxData[tid + 4])
				maxData[tid] = maxData[tid + 4];

			if(minData[tid]>minData[tid + 4])
				minData[tid] = minData[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			if(maxData[tid]<maxData[tid + 2])
				maxData[tid] = maxData[tid + 2];

			if(minData[tid]>minData[tid + 2])
				minData[tid] = minData[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			if(maxData[tid]<maxData[tid + 1])
				maxData[tid] = maxData[tid + 1];

			if(minData[tid]>minData[tid + 1])
				minData[tid] = minData[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
	{
		minArrayOut[blockIdx.x] = minData[0];
		maxArrayOut[blockIdx.x] = maxData[0];
	}
}

__global__ void cudaPolyTransferFuncImage( CudaImageContainer imageIn, CudaImageContainer imageOut, double a, double b, double c,
										  DevicePixelType minPixelValue, DevicePixelType maxPixelValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		double pixVal = (double)imageIn[coordinate] / maxPixelValue;
		double multiplier = a*pixVal*pixVal + b*pixVal + c;
		if (multiplier<0)
			multiplier = 0;
		if (multiplier>1)
			multiplier = 1;

		DevicePixelType newPixelVal = min((double)maxPixelValue,max((double)minPixelValue, multiplier*maxPixelValue));

		imageOut[coordinate] = newPixelVal;
	}
}

__global__ void cudaSumArray(CudaImageContainer arrayIn, double* arrayOut, size_t n)
{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013

	extern __shared__ double sdata[];

	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockDim.x*2 + tid;
	size_t gridSize = blockDim.x*2*gridDim.x;
	sdata[tid] = 0;

	while (i<n)
	{
		sdata[tid] = arrayIn[i];

		if (i+blockDim.x<n)
			sdata[tid] += arrayIn[i+blockDim.x];

		i += gridSize;
	}
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
			sdata[tid] += sdata[tid + 1024];
		__syncthreads();
	}
	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads(); 
	}
	if (blockDim.x >= 128) 
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			sdata[tid] += sdata[tid + 32];
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			sdata[tid] += sdata[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			sdata[tid] += sdata[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			sdata[tid] += sdata[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			sdata[tid] += sdata[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			sdata[tid] += sdata[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
		arrayOut[blockIdx.x] = sdata[0];
}

__global__ void cudaRuduceImage( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<double> hostReductions )
{
	DeviceVec<double> reductions = hostReductions;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageOut.getDims())
	{
		double val = 0;
		DeviceVec<size_t> mins, maxs;
		mins.x = coordinate.x*reductions.x;
		maxs.x = min(mins.x+reductions.x,(double)imageIn.getWidth());
		mins.y = coordinate.y*reductions.y;
		maxs.y = min(mins.y+reductions.y,(double)imageIn.getHeight());
		mins.z = coordinate.z*reductions.z;
		maxs.z = min(mins.z+reductions.z,(double)imageIn.getDepth());

		DeviceVec<size_t> currCorrd;
		for (currCorrd.z=mins.z; currCorrd.z<maxs.z; ++currCorrd.z)
		{
			for (currCorrd.y=mins.y; currCorrd.y<maxs.y; ++currCorrd.y)
			{
				for (currCorrd.x=mins.x; currCorrd.x<maxs.x; ++currCorrd.x)
				{
					val += (float)imageIn[currCorrd];
				}
			}
		}

		imageOut[coordinate] = val/(maxs-mins).product();
	}
}

__global__ void cudaMaximumIntensityProjection( CudaImageContainer imageIn, CudaImageContainer imageOut )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims() && coordinate.z==0)
	{
		DevicePixelType maxVal = 0;
		for (; coordinate.z<imageIn.getDepth(); ++coordinate.z)
		{
			if (maxVal<imageIn[coordinate])
			{
				maxVal = imageIn[coordinate];
			}
		}

		coordinate.z = 0;
		imageOut[coordinate] = maxVal;
	}
}

__global__ void cudaGetROI( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostStartPos, Vec<size_t> hostNewSize )
{
	DeviceVec<size_t> newSize = hostNewSize;
	DeviceVec<size_t> startPos = hostStartPos;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate>=startPos && coordinate<startPos+newSize && coordinate<imageIn.getDims())
	{
		imageOut[coordinate-startPos] = imageIn[coordinate];
	}
}

__global__ void cudaPow( CudaImageContainer imageIn, CudaImageContainer imageOut, double p )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
		imageOut[coordinate] = pow((double)imageIn[coordinate],p);
}

__global__ void cudaUnmixing( const CudaImageContainer imageIn1, const CudaImageContainer imageIn2, CudaImageContainer imageOut1,
							 Vec<size_t> hostKernelDims, DevicePixelType minPixelValue, DevicePixelType maxPixelValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		double meanIm1 = 0;
		double meanIm2 = 0;		int kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn1.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn1.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn1.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					meanIm1 += imageIn1[curCoordIm];
					meanIm2 += imageIn2[curCoordIm];
					++kernelVolume;
				}
			}
		}

		meanIm1 /= kernelVolume;
		meanIm2 /= kernelVolume;

		if (meanIm1 < meanIm2)
		{
			imageOut1[coordinate] = min(maxPixelValue, max(imageIn1[coordinate]-imageIn2[coordinate],minPixelValue));
		}
		else 
		{
			imageOut1[coordinate] = imageIn1[coordinate];
		}	
	}
}

__global__ void cudaMask( const CudaImageContainer imageIn1, const CudaImageContainer imageIn2, CudaImageContainer imageOut,
						 DevicePixelType threshold )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		DevicePixelType val=0;

		if (imageIn2[coordinate] <= threshold)
			val = imageIn1[coordinate];

		imageOut[coordinate] = val;
	}
}