#pragma once
#include "Vec.h"

#include <limits.h>

Vec<size_t> createGaussianKernel(Vec<float> sigma, float** kernel, Vec<int>& iterations);
Vec<size_t> createLoGKernel(Vec<float> sigma, float** kernelOut, size_t& kernSize);
Vec<size_t> createGaussianKernelFull(Vec<float> sigma, float** kernelOut, Vec<size_t> maxKernelSize = Vec<size_t>(ULLONG_MAX));
