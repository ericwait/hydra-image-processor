#pragma once

#include "../Cuda/ImageView.h"
#include "../Cuda/Vec.h"

#include <mex.h>

ImageOwner<float> getKernel(const mxArray* mexKernel);
