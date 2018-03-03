#include "CudaUtilities.h"

void calcBlockThread(const Vec<size_t>& dims, size_t maxThreads, dim3 &blocks, dim3 &threads)
{
	if (dims.z <= 1)
	{
		if (dims.y <= 1)
		{
			if (dims.x < maxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = 1;
				threads.z = 1;
			} 
			else
			{
				threads.x = (unsigned int)maxThreads;
				threads.y = 1;
				threads.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y < maxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = (unsigned int)dims.y;
				threads.z = 1;
			} 
			else
			{
				int dim = (unsigned int)sqrt((double)maxThreads);

				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z < maxThreads)
		{
			threads.x = (unsigned int)dims.x;
			threads.y = (unsigned int)dims.y;
			threads.z = (unsigned int)dims.z;
		}
		else
		{
			unsigned long index;
			_BitScanReverse(&index,unsigned long(maxThreads));

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
			sprintf_s(buff,"Out of CUDA Memory!\nNeed: %zu\nHave: %zu\n",needed,free);
			throw std::runtime_error(buff);
		}
		return false;
	}
	return true;
}
