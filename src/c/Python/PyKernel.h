#pragma once

#include "../Cuda/ImageContainer.h"
#include "../Cuda/Vec.h"

#include "PyIncludes.h"

ImageContainer<float> getKernel(PyArrayObject* kernel);
