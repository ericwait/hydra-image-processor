#pragma once
#include "Vec.h"

#include <limits.h>
#include <vector>

float* createGaussianKernel(Vec<double> sigmas, Vec<size_t>& dimsOut);
float* createLoG_GausKernels(Vec<double> sigmas, Vec<size_t>& dimsOut);
