#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>

float* createEllipsoidKernel(Vec<size_t> radii, Vec<size_t>& kernelDims);

int calcOtsuThreshold(const double* normHistogram, int numBins);

template <class PixelType>
PixelType* setUpOutIm(Vec<size_t> dims, PixelType** imageOut)
{
	DevicePixelType* imOut;
	if (imageOut==NULL)
		imOut = new DevicePixelType[dims.product()];
	else
		imOut = *imageOut;

	return imOut;
}
