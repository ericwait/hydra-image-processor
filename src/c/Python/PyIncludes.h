#pragma once

#include <Python.h>

// Make sure that Numpy symbols don't get re-imported in multiple compilation units
#ifndef NUMPY_IMPORT_MODULE
	#define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL HIP_ARRAY_API
#include <numpy/arrayobject.h>

#include <py3c.h>