#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>

double* createCircleKernel(int rad, Vec<unsigned int>& kernelDims);

int calcOtsuThreshold(const double* normHistogram, int numBins);