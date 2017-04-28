#include "CudaUtilities.cuh"

void calcBlockThread(const Vec<size_t>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads,
					 size_t maxThreads/*=std::numeric_limits<size_t>::max()*/)
{
	size_t mxThreads = MIN(prop.maxThreadsPerBlock,maxThreads);
	if (dims.z <= 1)
	{
		if (dims.y <= 1)
		{
			if (dims.x < mxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = 1;
				threads.z = 1;
			} 
			else
			{
				threads.x = (unsigned int)mxThreads;
				threads.y = 1;
				threads.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y < mxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = (unsigned int)dims.y;
				threads.z = 1;
			} 
			else
			{
				int dim = (unsigned int)sqrt((double)mxThreads);

				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z < mxThreads)
		{
			threads.x = (unsigned int)dims.x;
			threads.y = (unsigned int)dims.y;
			threads.z = (unsigned int)dims.z;
		}
		else
		{
			unsigned long index;
			_BitScanReverse(&index,unsigned long(mxThreads));

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

Vec<size_t> createLoGKernel(Vec<float> sigma, float** kernelOut, Vec<int>& iterations)
{
	const double PI = std::atan(1.0)*4;

	Vec<size_t> kernelDims(1, 1, 1);
	iterations = Vec<int>(1, 1, 1);

	if((sigma.x+sigma.y+sigma.z)*10>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		iterations.x = (int)MAX(1.0f, ceil(100.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM)));
		iterations.y = (int)MAX(1.0f, ceil(100.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM)));
		iterations.z = (int)MAX(1.0f, ceil(100.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM)));

		//TODO: Optimize iterations per dim
		sigma.x = sigma.x/sqrt((float)iterations.x);
		sigma.y = sigma.y/sqrt((float)iterations.y);
		sigma.z = sigma.z/sqrt((float)iterations.z);
	}

	kernelDims.x = (size_t)MAX(1.0f, (10*sigma.x));
	kernelDims.y = (size_t)MAX(1.0f, (10*sigma.y));
	kernelDims.z = (size_t)MAX(1.0f, (10*sigma.z));

	kernelDims.x = (kernelDims.x%2==0) ? (kernelDims.x+1) : (kernelDims.x);
	kernelDims.y = (kernelDims.y%2==0) ? (kernelDims.y+1) : (kernelDims.y);
	kernelDims.z = (kernelDims.z%2==0) ? (kernelDims.z+1) : (kernelDims.z);

	Vec<float> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	*kernelOut = new float[kernelDims.sum()];
	float* kernel = *kernelOut;

	double piPow = 2.0;
	double sigmaDem = 0.0;
	double sigmaSub = 0.0;
	int numDim = 0;

	if(sigma.x!=0)
	{
		++numDim;
		sigmaSub = sigmaSub-1.0/sigma.x;
		if(sigma.y!=0)
		{
			++numDim;
			sigmaSub = sigmaSub-1.0/sigma.y;
			if(sigma.z!=0)
			{
				++numDim;
				sigmaSub = sigmaSub-1.0/sigma.z;
				sigmaDem = sigma.product();
				piPow = 4.0;
			}
			else
			{
				sigmaDem = SQR(sigma.x) * SQR(sigma.y);
			}
		}
		else
		{
			if(sigma.z!=0)
			{
				++numDim;
				sigmaSub = sigmaSub-1.0/sigma.z;
				sigmaDem = SQR(sigma.x) * SQR(sigma.z);
			}
			else
			{
				sigmaDem = SQR(sigma.x);
			}
		}
	}
	else
	{
		if(sigma.y!=0)
		{
			++numDim;
			sigmaSub = sigmaSub-1.0/sigma.y;
			if(sigma.z!=0)
			{
				++numDim;
				sigmaSub = sigmaSub-1.0/sigma.z;
				sigmaDem = SQR(sigma.y) * SQR(sigma.z);
			}
			else
			{
				sigmaDem = SQR(sigma.y);
			}
		} 
		else
		{
			if(sigma.z!=0)
			{
				++numDim;
				sigmaSub = sigmaSub-1.0/sigma.z;
				sigmaDem = SQR(sigma.z);
			} 
			else
			{
				std::runtime_error("One dimension has to have a non-zero sigma!");
			}
		}
	}

	double sigmaGPwr = (numDim==3) ? (4) : (2);
	double sigmaEPwr = (numDim==3) ? (2) : (0);
	Vec<double> sigmaG = sigma.pwr(sigmaGPwr);
	Vec<double> sigmaE = Vec<double>(sigma).pwr(sigmaEPwr)*2.0;
	double dem = pow((2*PI), piPow) * sigmaDem;

	for(int i = 0; i<3; ++i)
	{
		size_t indStride = 0;
		for(int j = 0; j<i; ++j)
		{
			indStride += kernelDims.e[j];
		}

		if(sigma.e[i]==0)
		{
			kernel[indStride] = 1.0f;
		} else
		{
			for(int j = 0; j<kernelDims.e[i]; ++j)
			{
				double jSqr = SQR(j-mid.e[i]);// make this a coordinate based on a zero mean
				double kernelVal = (jSqr/sigmaG.e[i]);
				kernelVal += sigmaSub;
				kernelVal *= exp(-(jSqr/(sigmaE.e[i])));
				kernelVal /= dem;

				kernel[j+indStride] = (float)kernelVal;
			}
		}
	}

	return kernelDims;
}

Vec<size_t> createGaussianKernelFull(Vec<float> sigma, float** kernelOut, Vec<size_t> maxKernelSize)
{
    Vec<size_t> kernelDims = Vec<size_t>(sigma.clamp(Vec<float>(1.0f), std::numeric_limits<float>::max()));

    for(float numStd = 3.0f; numStd>1.0f; numStd -= 0.2f)
    {
        if(sigma.product()*numStd<maxKernelSize.product())
        {
            kernelDims = sigma*numStd+0.9999f;
            break;
        }
    }

    kernelDims = kernelDims.clamp(Vec<size_t>(1), Vec<size_t>(std::numeric_limits<size_t>::max()));
    sigma = sigma.clamp(Vec<float>(0.1f), Vec<float>(std::numeric_limits<float>::max()));

    if(kernelDims.product()>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
    {
        kernelDims = Vec<size_t>(MAX_KERNEL_DIM, MAX_KERNEL_DIM, MAX_KERNEL_DIM);
    }

    Vec<float> center = (kernelDims-1.0f)/2.0f;

    *kernelOut = new float[kernelDims.product()];
    float* kernel = *kernelOut;

    float total = 0.0f;
    Vec<float> pos(0, 0, 0);
    
    Vec<float> denominator = SQR(sigma)*2;
    for(pos.z = 0; pos.z<kernelDims.z; ++pos.z)
    {
        for(pos.y = 0; pos.y<kernelDims.y; ++pos.y)
        {
            for(pos.x = 0; pos.x<kernelDims.x; ++pos.x)
            {
                Vec<float> mahal = SQR(center-pos)/denominator;
                kernel[kernelDims.linearAddressAt(pos)] = exp(-(mahal.sum()));
                total += kernel[kernelDims.linearAddressAt(pos)];
            }
        }
    }

    for(int i = 0; i<kernelDims.product(); ++i)
        kernel[i] /= total;

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
			sprintf_s(buff,"Out of CUDA Memory!\nNeed: %zu\nHave: %zu\n",needed,free);
			throw std::runtime_error(buff);
		}
		return false;
	}
	return true;
}
