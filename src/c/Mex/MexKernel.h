#pragma once

#include "../Cuda/ImageView.h"
#include "../Cuda/Vec.h"

#include "MexIncludes.h"

ImageOwner<float> getKernel(const mxArray* mexKernel);
