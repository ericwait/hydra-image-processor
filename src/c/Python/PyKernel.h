#pragma once

#include "../Cuda/ImageView.h"
#include "../Cuda/Vec.h"

#include "PyIncludes.h"

ImageOwner<float> getKernel(PyArrayObject* kernel);
