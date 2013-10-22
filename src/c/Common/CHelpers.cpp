#include "CHelpers.h"

double* createCircleKernel(int rad, Vec<unsigned int>& kernelDims)
{
	kernelDims.x = rad%2==0 ? rad+1 : rad;
	kernelDims.y = rad%2==0 ? rad+1 : rad;
	kernelDims.z = rad%2==0 ? rad+1 : rad;

	double* kernel = new double[kernelDims.product()];
	memset(kernel,0,sizeof(double)*kernelDims.product());

	Vec<unsigned int> mid;
	mid.x = kernelDims.x/2+1;
	mid.y = kernelDims.y/2+1;
	mid.z = kernelDims.z/2+1;

	Vec<unsigned int> cur(0,0,0);
	for (cur.x=0; cur.x < kernelDims.x ; ++cur.x)
	{
		for (cur.y=0; cur.y < kernelDims.y ; ++cur.y)
		{
			for (cur.z=0; cur.z < kernelDims.z ; ++cur.z)
			{
				if (cur.EuclideanDistanceTo(mid)<=rad)
					kernel[kernelDims.linearAddressAt(cur)] = 1;
			}
		}
	}

	return kernel;
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