#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>
#define MAX_KERNAL_DIM (15)

double* createCircleKernal(int rad, Vec<int>& kernalDims)
{
	kernalDims.x = rad%2==0 ? rad+1 : rad;
	kernalDims.y = rad%2==0 ? rad+1 : rad;
	kernalDims.z = rad%2==0 ? rad+1 : rad;

	double* kernal = new double[kernalDims.x*kernalDims.y*kernalDims.z];
	memset(kernal,0,sizeof(double)*kernalDims.x*kernalDims.y*kernalDims.z);

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
					kernal[cur.x+cur.y*kernalDims.x+cur.z*kernalDims.y*kernalDims.x] = 1;
			}
		}
	}

	return kernal;
}

double* createGaussianKernal(Vec<double> sigma, Vec<int>& kernalDims)
{
	kernalDims.x = (int)(3*sigma.x);
	kernalDims.y = (int)(3*sigma.y);
	kernalDims.z = (int)(3*sigma.z);

	kernalDims.x = kernalDims.x%2==0 ? kernalDims.x+1 : kernalDims.x;
	kernalDims.y = kernalDims.y%2==0 ? kernalDims.y+1 : kernalDims.y;
	kernalDims.z = kernalDims.z%2==0 ? kernalDims.z+1 : kernalDims.z;

	double* kernal = new double[kernalDims.x*kernalDims.y*kernalDims.z];

	Vec<int> mid;
	mid.x = kernalDims.x/2;
	mid.y = kernalDims.y/2;
	mid.z = kernalDims.z/2;

	Vec<int> cur(0,0,0);
	Vec<double> gaus;
	double total = 0.0;
	for (cur.x=0; cur.x < kernalDims.x ; ++cur.x)
	{
		for (cur.y=0; cur.y < kernalDims.y ; ++cur.y)
		{
			for (cur.z=0; cur.z < kernalDims.z ; ++cur.z)
			{
				gaus.x = exp(-((mid.x-cur.x)*(mid.x-cur.x)) / (2*sigma.x*sigma.x));
				gaus.y = exp(-((mid.y-cur.y)*(mid.y-cur.y)) / (2*sigma.y*sigma.y));
				gaus.z = exp(-((mid.z-cur.z)*(mid.z-cur.z)) / (2*sigma.z*sigma.z));

				total += kernal[cur.x+cur.y*kernalDims.x+cur.z*kernalDims.y*kernalDims.x] = gaus.x*gaus.y*gaus.z;
			}
		}
	}

	for (cur.x=0; cur.x < kernalDims.x ; ++cur.x)
		for (cur.y=0; cur.y < kernalDims.y ; ++cur.y)
			for (cur.z=0; cur.z < kernalDims.z ; ++cur.z)
				kernal[cur.x+cur.y*kernalDims.x+cur.z*kernalDims.y*kernalDims.x] /= total;

	return kernal;
}