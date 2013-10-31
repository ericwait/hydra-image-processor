#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>

double* createEllipsoidKernel(Vec<size_t> radii, Vec<size_t>& kernelDims);