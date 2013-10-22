#include "CudaUtilities.cuh"

void calcBlockThread(const Vec<unsigned int>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads)
{
	if (dims.z==1)
	{
		if (dims.y==1)
		{
			if ((int)dims.x<prop.maxThreadsPerBlock)
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

				blocks.x = (int)ceil((float)dims.x/prop.maxThreadsPerBlock);
				blocks.y = 1;
				blocks.z = 1;
			}
		}
		else
		{
			if ((int)(dims.x*dims.y)<prop.maxThreadsPerBlock)
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
				int dim = (int)sqrt((double)prop.maxThreadsPerBlock);

				threads.x = dim;
				threads.y = dim;
				threads.z = 1;

				blocks.x = (int)ceil((float)dims.x/dim);
				blocks.y = (int)ceil((float)dims.y/dim);
				blocks.z = 1;
			}
		}
	}
	else
	{
		if((int)(dims.x*dims.y*dims.z)<prop.maxThreadsPerBlock)
		{
			threads.x = dims.x;
			threads.y = dims.y;
			threads.z = dims.z;

			blocks.x = 1;
			blocks.y = 1;
			blocks.z = 1;
		}
		else
		{
			int dim = (int)pow((float)prop.maxThreadsPerBlock,1/3.0f);
			float extra = (float)(prop.maxThreadsPerBlock-dim*dim*dim)/(dim*dim);

			threads.x = dim + (int)extra;
			threads.y = dim;
			threads.z = dim;

			blocks.x = (unsigned int)ceil((float)dims.x/threads.x);
			blocks.y = (unsigned int)ceil((float)dims.y/threads.y);
			blocks.z = (unsigned int)ceil((float)dims.z/threads.z);
		}
	}
}

template<typename KernelType>
Vec<unsigned int> createGaussianKernel(Vec<float> sigma, float* kernel, int& iterations)
{
	Vec<unsigned int> kernelDims(1,1,1);
	iterations = 1;

	if (sigma.product()*27>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		Vec<int> minIterations;
		minIterations.x = (int)ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM));
		minIterations.y = (int)ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM));
		minIterations.z = (int)ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM));

		iterations = minIterations.maxValue();
		sigma = sigma/sqrt((float)iterations);
	}

	kernelDims.x = (unsigned int)(3*sigma.x);
	kernelDims.y = (unsigned int)(3*sigma.y);
	kernelDims.z = (unsigned int)(3*sigma.z);

	kernelDims.x = kernelDims.x%2==0 ? kernelDims.x+1 : kernelDims.x;
	kernelDims.y = kernelDims.y%2==0 ? kernelDims.y+1 : kernelDims.y;
	kernelDims.z = kernelDims.z%2==0 ? kernelDims.z+1 : kernelDims.z;

	Vec<unsigned int> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	Vec<unsigned int> cur(0,0,0);
	Vec<float> gaus;
	float total = 0.0;
	for (cur.x=0; cur.x<kernelDims.x ; ++cur.x)
	{
		for (cur.y=0; cur.y<kernelDims.y ; ++cur.y)
		{
			for (cur.z=0; cur.z<kernelDims.z ; ++cur.z)
			{
				gaus.x = exp(-(int)(SQR(mid.x-cur.x)) / (2*SQR(sigma.x)));
				gaus.y = exp(-(int)(SQR(mid.y-cur.y)) / (2*SQR(sigma.y)));
				gaus.z = exp(-(int)(SQR(mid.z-cur.z)) / (2*SQR(sigma.z)));

				total += kernel[kernelDims.linearAddressAt(cur)] = (float)gaus.product();
			}
		}
	}

	for (cur.x=0; cur.x < kernelDims.x ; ++cur.x)
		for (cur.y=0; cur.y < kernelDims.y ; ++cur.y)
			for (cur.z=0; cur.z < kernelDims.z ; ++cur.z)
				kernel[kernelDims.linearAddressAt(cur)] /= total;

	return kernelDims;
}

Vec<unsigned int> createGaussianKernel(Vec<float> sigma, float* kernel, Vec<int>& iterations)
{
	Vec<unsigned int> kernelDims(1,1,1);
	iterations = Vec<int>(1,1,1);

	if ((sigma.x+sigma.y+sigma.z)*3>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		iterations.x = (int)ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM));
		iterations.y = (int)ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM));
		iterations.z = (int)ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM));

		//TODO: Optimize iterations per dim
		sigma.x = sigma.x/sqrt((float)iterations.x);
		sigma.y = sigma.y/sqrt((float)iterations.y);
		sigma.z = sigma.z/sqrt((float)iterations.z);
	}

	kernelDims.x = (unsigned int)(3*sigma.x);
	kernelDims.y = (unsigned int)(3*sigma.y);
	kernelDims.z = (unsigned int)(3*sigma.z);

	kernelDims.x = kernelDims.x%2==0 ? kernelDims.x+1 : kernelDims.x;
	kernelDims.y = kernelDims.y%2==0 ? kernelDims.y+1 : kernelDims.y;
	kernelDims.z = kernelDims.z%2==0 ? kernelDims.z+1 : kernelDims.z;

	Vec<unsigned int> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	float total = 0.0;
	for (unsigned int x=0; x<kernelDims.x ; ++x)
		total += kernel[x] =  exp(-(int)(SQR(mid.x-x)) / (2*SQR(sigma.x)));
	for (unsigned int x=0; x<kernelDims.x ; ++x)
		kernel[x] /= total;

	total = 0.0;
	for (unsigned int y=0; y<kernelDims.y ; ++y)
		total += kernel[y+kernelDims.x] = exp(-(int)(SQR(mid.y-y)) / (2*SQR(sigma.y)));
	for (unsigned int y=0; y < kernelDims.y ; ++y)
		kernel[y+kernelDims.x] /= total;

	total = 0.0;
	for (unsigned int z=0; z<kernelDims.z ; ++z)
		total += kernel[z+kernelDims.x+kernelDims.y] = exp(-(int)(SQR(mid.z-z)) / (2*SQR(sigma.z)));
	for (unsigned int z=0; z < kernelDims.z ; ++z)
		kernel[z+kernelDims.x+kernelDims.y] /= total;


	return kernelDims;
}