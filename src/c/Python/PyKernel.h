#pragma once

#include "../Cuda/ImageContainer.h"
#include "../Cuda/Vec.h"

#include "PyIncludes.h"

ImageOwner<float> getKernel(PyArrayObject* kernel);
