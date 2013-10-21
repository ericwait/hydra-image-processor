#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>
#define MAX_KERNAL_DIM (25)

double* createCircleKernal(int rad, Vec<int>& kernalDims)
{
	kernalDims.x = rad%2==0 ? rad+1 : rad;
	kernalDims.y = rad%2==0 ? rad+1 : rad;
	kernalDims.z = rad%2==0 ? rad+1 : rad;

	double* kernal = new double[kernalDims.product()];
	memset(kernal,0,sizeof(double)*kernalDims.product());

	Vec<int> mid;
	mid.x = kernalDims.x/2+1;
	mid.y = kernalDims.y/2+1;
	mid.z = kernalDims.z/2+1;

	Vec<int> cur(0,0,0);
	for (cur.x=0; cur.x < kernalDims.x ; ++cur.x)
	{
		for (cur.y=0; cur.y < kernalDims.y ; ++cur.y)
		{
			for (cur.z=0; cur.z < kernalDims.z ; ++cur.z)
			{
				if (cur.EuclideanDistanceTo(mid)<=rad)
					kernal[kernalDims.linearAddressAt(cur)] = 1;
			}
		}
	}

	return kernal;
}

template<typename KernalType>
Vec<unsigned int> createGaussianKernal(Vec<float> sigma, KernalType* kernal, int& iterations)
{
	Vec<unsigned int> kernalDims(1,1,1);
	iterations = 1;

	if (sigma.product()*27>MAX_KERNAL_DIM*MAX_KERNAL_DIM*MAX_KERNAL_DIM)
	{
		Vec<int> minIterations;
		minIterations.x = (int)ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNAL_DIM));
		minIterations.y = (int)ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNAL_DIM));
		minIterations.z = (int)ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNAL_DIM));

		iterations = minIterations.maxValue();
		sigma = sigma/sqrt((float)iterations);
	}

	kernalDims.x = (unsigned int)(3*sigma.x);
	kernalDims.y = (unsigned int)(3*sigma.y);
	kernalDims.z = (unsigned int)(3*sigma.z);

	kernalDims.x = kernalDims.x%2==0 ? kernalDims.x+1 : kernalDims.x;
	kernalDims.y = kernalDims.y%2==0 ? kernalDims.y+1 : kernalDims.y;
	kernalDims.z = kernalDims.z%2==0 ? kernalDims.z+1 : kernalDims.z;

	Vec<unsigned int> mid;
	mid.x = kernalDims.x/2;
	mid.y = kernalDims.y/2;
	mid.z = kernalDims.z/2;

	Vec<unsigned int> cur(0,0,0);
	Vec<float> gaus;
	float total = 0.0;
	for (cur.x=0; cur.x<kernalDims.x ; ++cur.x)
	{
		for (cur.y=0; cur.y<kernalDims.y ; ++cur.y)
		{
			for (cur.z=0; cur.z<kernalDims.z ; ++cur.z)
			{
				gaus.x = exp(-(int)(SQR(mid.x-cur.x)) / (2*SQR(sigma.x)));
				gaus.y = exp(-(int)(SQR(mid.y-cur.y)) / (2*SQR(sigma.y)));
				gaus.z = exp(-(int)(SQR(mid.z-cur.z)) / (2*SQR(sigma.z)));

				total += kernal[kernalDims.linearAddressAt(cur)] = (float)gaus.product();
			}
		}
	}

	for (cur.x=0; cur.x < kernalDims.x ; ++cur.x)
		for (cur.y=0; cur.y < kernalDims.y ; ++cur.y)
			for (cur.z=0; cur.z < kernalDims.z ; ++cur.z)
				kernal[kernalDims.linearAddressAt(cur)] /= total;

	return kernalDims;
}

template<typename KernalType>
Vec<unsigned int> createGaussianKernal(Vec<float> sigma, KernalType* kernal, Vec<int>& iterations)
{
	Vec<unsigned int> kernalDims(1,1,1);
	iterations = Vec<int>(1,1,1);

	if ((sigma.x+sigma.y+sigma.z)*3>MAX_KERNAL_DIM*MAX_KERNAL_DIM*MAX_KERNAL_DIM)
	{
		iterations.x = (int)ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNAL_DIM));
		iterations.y = (int)ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNAL_DIM));
		iterations.z = (int)ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNAL_DIM));

		//TODO: Optimize iterations per dim
		sigma.x = (float)(sigma.x/sqrt((float)iterations.x));
		sigma.y = (float)(sigma.y/sqrt((float)iterations.y));
		sigma.z = (float)(sigma.z/sqrt((float)iterations.z));
	}

	kernalDims.x = (unsigned int)(3*sigma.x);
	kernalDims.y = (unsigned int)(3*sigma.y);
	kernalDims.z = (unsigned int)(3*sigma.z);

	kernalDims.x = kernalDims.x%2==0 ? kernalDims.x+1 : kernalDims.x;
	kernalDims.y = kernalDims.y%2==0 ? kernalDims.y+1 : kernalDims.y;
	kernalDims.z = kernalDims.z%2==0 ? kernalDims.z+1 : kernalDims.z;

	Vec<unsigned int> mid;
	mid.x = kernalDims.x/2;
	mid.y = kernalDims.y/2;
	mid.z = kernalDims.z/2;

	float total = 0.0;
	for (unsigned int x=0; x<kernalDims.x ; ++x)
		total += kernal[x] =  exp(-(int)(SQR(mid.x-x)) / (2*SQR(sigma.x)));
	for (unsigned int x=0; x<kernalDims.x ; ++x)
		kernal[x] /= total;

	total = 0.0;
	for (unsigned int y=0; y<kernalDims.y ; ++y)
		total += kernal[y+kernalDims.x] = exp(-(int)(SQR(mid.y-y)) / (2*SQR(sigma.y)));
	for (unsigned int y=0; y < kernalDims.y ; ++y)
		kernal[y+kernalDims.x] /= total;

	total = 0.0;
	for (unsigned int z=0; z<kernalDims.z ; ++z)
		total += kernal[z+kernalDims.x+kernalDims.y] = exp(-(int)(SQR(mid.z-z)) / (2*SQR(sigma.z)));
	for (unsigned int z=0; z < kernalDims.z ; ++z)
		kernal[z+kernalDims.x+kernalDims.y] /= total;
				

	return kernalDims;
}

int calcOtsuThreshold(const double* normHistogram, int numBins)
{
	//code modified from http://www.dandiggins.co.uk/arlib-9.html
	double totalMean = 0.0f;
	for (int i=0; i<numBins; ++i)
		totalMean += i*normHistogram[i];

	double class1Prob=0, class1Mean=0, temp1, curThresh;
	double bestThresh = 0.0;
	int bestIndex = 0;
	for (int i=0; i<numBins; ++i)
	{
		class1Prob += normHistogram[i];
		class1Mean += i * normHistogram[i];

		temp1 = totalMean * class1Prob - class1Mean;
		curThresh = (temp1*temp1) / (class1Prob*(1.0f-class1Prob));

		if(curThresh>bestThresh)
		{
			bestThresh = curThresh;
			bestIndex = i;
		}
	}

	return bestIndex;
}