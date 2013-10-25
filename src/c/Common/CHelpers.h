#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>

double* createEllipsoidKernel(Vec<unsigned int> radii, Vec<unsigned int>& kernelDims);