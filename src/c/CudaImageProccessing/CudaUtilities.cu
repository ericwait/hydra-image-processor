#include "CudaUtilities.h"

void calcBlockThread(const Vec<int>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads)
{
	if (dims.z==1)
	{
		if (dims.y==1)
		{
			if (dims.x<prop.maxThreadsPerBlock)
			{
				threads.x = dims.x;
				threads.y = 1;
				threads.z = 1;
				blocks.x = 1;
				blocks.y = 1;
				blocks.z = 1;
			} 
			else
			{
				threads.x = prop.maxThreadsPerBlock;
				threads.y = 1;
				threads.z = 1;
				blocks.x = ceil((float)dims.x/prop.maxThreadsPerBlock);
				blocks.y = 1;
				blocks.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y<prop.maxThreadsPerBlock)
			{
				threads.x = dims.x;
				threads.y = dims.y;
				threads.z = 1;
				blocks.x = 1;
				blocks.y = 1;
				blocks.z = 1;
			} 
			else
			{
				int dim = sqrt((float)prop.maxThreadsPerBlock);
				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
				blocks.x = ceil((float)dims.x/dim);
				blocks.y = ceil((float)dims.y/dim);
				blocks.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z < prop.maxThreadsPerBlock)
		{
			blocks.x = 1;
			blocks.y = 1;
			blocks.z = 1;
			threads.x = dims.x;
			threads.y = dims.y;
			threads.z = dims.z;
		}
		else
		{
			int dim = (int)pow((float)prop.maxThreadsPerBlock,1/3.0f);
			int extra = (prop.maxThreadsPerBlock-dim*dim*dim)/(dim*dim);
			threads.x = dim + extra;
			threads.y = dim;
			threads.z = dim;

			blocks.x = (unsigned int)ceil((float)dims.x/threads.x);
			blocks.y = (unsigned int)ceil((float)dims.y/threads.y);
			blocks.z = (unsigned int)ceil((float)dims.z/threads.z);
		}
	}
}