#pragma once

#include <Python.h>

// Make sure that Numpy symbols don't get re-imported in multiple compilation units
#ifndef NUMPY_IMPORT_MODULE
#define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL HIP_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <py3c.h>
#include <cstddef>
#include <cstdint>

namespace Script
{
	typedef npy_intp		DimType;
	typedef PyObject		ObjectType;
	typedef PyArrayObject	ArrayType;

	// Simple template-specialization map for C++ to Python types
	BEGIN_TYPE_MAP(NPY_TYPES)
		TYPE_MAPPING(bool, NPY_BOOL)
		TYPE_MAPPING(int8_t, NPY_INT8)
		TYPE_MAPPING(int16_t, NPY_INT16)
		TYPE_MAPPING(int32_t, NPY_INT32)
		TYPE_MAPPING(uint8_t, NPY_UINT8)
		TYPE_MAPPING(uint16_t, NPY_UINT16)
		TYPE_MAPPING(uint32_t, NPY_UINT32)
		TYPE_MAPPING(float, NPY_FLOAT)
		TYPE_MAPPING(double, NPY_DOUBLE)
	END_TYPE_MAP(NPY_TYPES)

	template <typename T> ArrayType* createArray(int ndim, DimType* dims)
	{
		return ((ArrayType*) PyArray_SimpleNew(ndim, dims, ID_FROM_TYPE(T)));
	}

	// TODO: Figure out if we can do this without the const-casting?
	namespace ArrayInfo
	{
		inline std::size_t	getNDims(const ArrayType* im) { return PyArray_NDIM(im); }
		inline DimType* getDims(const ArrayType* im) { return PyArray_DIMS(const_cast<ArrayType*>(im)); }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*) PyArray_DATA(const_cast<ArrayType*>(im)); }
	};

	// Some Python-specific converters
	bool pyobjToVec(ObjectType* list_array, Vec<double>& outVec);

	bool pylistToVec(ObjectType* list, Vec<double>& outVec);
	bool pyarrayToVec(ArrayType* ar, Vec<double>& outVec);

	template <typename T> PyObject* fromNumeric(T val){ return nullptr; }
	template <> inline PyObject* fromNumeric(bool val) { return PyBool_FromLong(val); }
	template <> inline PyObject* fromNumeric(uint8_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(uint16_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(int16_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(uint32_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(int32_t val) { return PyLong_FromLong(val); }
	template <> inline PyObject* fromNumeric(float val) { return PyFloat_FromDouble(val); }
	template <> inline PyObject* fromNumeric(double val) { return PyFloat_FromDouble(val); }
};
