#include "CHelpers.h"

double* createEllipsoidKernel(Vec<size_t> radii, Vec<size_t>& kernelDims)
{
	kernelDims.x = radii.x*2+1;
	kernelDims.y = radii.y*2+1;
	kernelDims.z = radii.z*2+1;

	double* kernel = new double[kernelDims.product()];
	memset(kernel,0,sizeof(double)*kernelDims.product());

	Vec<size_t> mid;
	mid.x = (kernelDims.x+1)/2;
	mid.y = (kernelDims.y+1)/2;
	mid.z = (kernelDims.z+1)/2;
	Vec<float> dimScale(1.0f/((float)SQR(radii.x)),1.0f/((float)SQR(radii.y)),1.0f/((float)SQR(radii.z)));

	Vec<size_t> cur(0,0,0);
	for (cur.z=0; cur.z<kernelDims.z ; ++cur.z)
	{
		for (cur.y=0; cur.y<kernelDims.y ; ++cur.y)
		{
			for (cur.x=0; cur.x<kernelDims.x ; ++cur.x)
			{
				if (dimScale.x*SQR(cur.x-mid.x)+dimScale.y*SQR(cur.y-mid.y)+dimScale.z*SQR(cur.z-mid.z)<=1)
					kernel[kernelDims.linearAddressAt(cur)] = 1.0f;
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