#pragma once
#include "Vec.h"

#include <limits.h>
#include <vector>

float* createGaussianKernel(Vec<double> sigmas, Vec<size_t>& dimsOut);
Vec<size_t> createLoGKernel(Vec<float> sigma, float** kernelOut, size_t& kernSize);
