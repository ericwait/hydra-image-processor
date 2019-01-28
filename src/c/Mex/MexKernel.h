#pragma once

#include "../Cuda/ImageContainer.h"
#include "../Cuda/Vec.h"

#include <mex.h>

ImageOwner<float> getKernel(const mxArray* mexKernel);
