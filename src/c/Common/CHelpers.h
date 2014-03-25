#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>

float* createEllipsoidKernel(Vec<size_t> radii, Vec<size_t>& kernelDims);

int calcOtsuThreshold(const double* normHistogram, int numBins);
