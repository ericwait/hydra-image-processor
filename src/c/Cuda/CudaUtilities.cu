#include "CudaUtilities.cuh"

void calcBlockThread(const Vec<size_t>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads)
{
	if (dims.z<=1)
	{
		if (dims.y<=1)
		{
			if (dims.x<(size_t)prop.maxThreadsPerBlock)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = 1;
				threads.z = 1;
			} 
			else
			{
				threads.x = prop.maxThreadsPerBlock;
				threads.y = 1;
				threads.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y<(size_t)prop.maxThreadsPerBlock)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = (unsigned int)dims.y;
				threads.z = 1;
			} 
			else
			{
				int dim = (unsigned int)sqrt((double)prop.maxThreadsPerBlock);

				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z<(size_t)prop.maxThreadsPerBlock)
		{
			threads.x = (unsigned int)dims.x;
			threads.y = (unsigned int)dims.y;
			threads.z = (unsigned int)dims.z;
		}
		else
		{
			unsigned long index;
			_BitScanReverse(&index,prop.maxThreadsPerBlock);

			int dim = index/3;
			threads.x = 1 << MAX(dim,(int)index - 2*dim);
			threads.y = 1 << dim;
			threads.z = 1 << MIN(dim,(int)index - 2*dim);
		}
	}

	blocks.x = (unsigned int)ceil((float)dims.x/threads.x);
	blocks.y = (unsigned int)ceil((float)dims.y/threads.y);
	blocks.z = (unsigned int)ceil((float)dims.z/threads.z);
}

Vec<size_t> createGaussianKernel(Vec<float> sigma, float** kernelOut, Vec<int>& iterations)
{
	Vec<size_t> kernelDims(1,1,1);
	iterations = Vec<int>(1,1,1);

	if ((sigma.x+sigma.y+sigma.z)*3>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		iterations.x = (int)MAX(1.0f,ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM)));
		iterations.y = (int)MAX(1.0f,ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM)));
		iterations.z = (int)MAX(1.0f,ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM)));

		//TODO: Optimize iterations per dim
		sigma.x = sigma.x/sqrt((float)iterations.x);
		sigma.y = sigma.y/sqrt((float)iterations.y);
		sigma.z = sigma.z/sqrt((float)iterations.z);
	}

	kernelDims.x = (size_t)MAX(1.0f,(3*sigma.x));
	kernelDims.y = (size_t)MAX(1.0f,(3*sigma.y));
	kernelDims.z = (size_t)MAX(1.0f,(3*sigma.z));

	kernelDims.x = (kernelDims.x%2==0) ? (kernelDims.x+1) : (kernelDims.x);
	kernelDims.y = (kernelDims.y%2==0) ? (kernelDims.y+1) : (kernelDims.y);
	kernelDims.z = (kernelDims.z%2==0) ? (kernelDims.z+1) : (kernelDims.z);

	Vec<size_t> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	*kernelOut = new float[kernelDims.sum()];
	float* kernel = *kernelOut;

	float total = 0.0;
	if (sigma.x==0)
	{
		kernel[0] = 1.0f;
	}
	else
	{
		for (size_t x=0; x<kernelDims.x ; ++x)
			total += kernel[x] =  exp(-(int)(SQR(mid.x-x)) / (2*SQR(sigma.x)));
		for (size_t x=0; x<kernelDims.x ; ++x)
			kernel[x] /= total;
	}

	total = 0.0;
	if (sigma.y==0)
	{
		kernel[kernelDims.x] = 1;
	}
	else
	{
		for (size_t y=0; y<kernelDims.y ; ++y)
			total += kernel[y+kernelDims.x] = exp(-(int)(SQR(mid.y-y)) / (2*SQR(sigma.y)));
		for (size_t y=0; y < kernelDims.y ; ++y)
			kernel[y+kernelDims.x] /= total;
	}

	total = 0.0;
	if (sigma.z==0)
	{
		kernel[kernelDims.x+kernelDims.y] = 1;
	}
	else
	{
		for (size_t z=0; z<kernelDims.z ; ++z)
			total += kernel[z+kernelDims.x+kernelDims.y] = exp(-(int)(SQR(mid.z-z)) / (2*SQR(sigma.z)));
		for (size_t z=0; z < kernelDims.z ; ++z)
			kernel[z+kernelDims.x+kernelDims.y] /= total;
	}

	return kernelDims;
}

size_t memoryAvailable(int device, size_t* totalOut/*=NULL*/)
{
	HANDLE_ERROR(cudaSetDevice(device));
	size_t free, total;
	HANDLE_ERROR(cudaMemGetInfo(&free,&total));

	if (totalOut!=NULL)
		*totalOut = total;

	return free;
}

bool checkFreeMemory(size_t needed, int device, bool throws/*=false*/)
{
	size_t free = memoryAvailable(device);
	if (needed>free)
	{
		if (throws)
		{
			char buff[255];
			sprintf_s(buff,"Out of CUDA Memory!\nNeed: %d\nHave: %d\n",needed,free);
			throw std::runtime_error(buff);
		}
		return false;
	}
	return true;
}
