#pragma once

#include "../Cuda/ImageContainer.h"
#include "../Cuda/Vec.h"

#include <mex.h>

ImageContainer<float> getKernel(const mxArray* mexKernel);
