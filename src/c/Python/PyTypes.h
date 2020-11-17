#pragma once

#include "PyIncludes.h"

#include <cstddef>
#include <cstdint>


#define CHECK_ARRAY_FLAGS(IM,FLAG) ((PyArray_FLAGS(IM) & (FLAG)) == (FLAG))

namespace Script
{
	typedef npy_intp		DimType;
	typedef PyObject		ObjectType;
	typedef PyArrayObject	ArrayType;

	struct ObjectDeleter { void operator() (ObjectType const* ptr){Py_XDECREF(ptr);} };
	struct ArrayDeleter { void operator() (ArrayType const* ptr){Py_XDECREF(ptr);} };

	typedef std::unique_ptr<ObjectType,ObjectDeleter>	GuardOutObjectPtr;
	typedef std::unique_ptr<ArrayType,ArrayDeleter>		GuardOutArrayPtr;

	// Simple template-specialization map for C++ to Python types
	BEGIN_TYPE_MAP(NPY_TYPES,Python)
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


	// Forward declaration of necessary functions
	inline DimInfo getDimInfo(const ArrayType* im);
	inline ImageDimensions makeImageDims(const DimInfo& info);


	template <typename... Args>
	inline void writeMsg(const char* fmt, Args&&... args)
	{
		PySys_WriteStdout(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	inline void warnMsg(const char* fmt, Args&&... args)
	{
		PyErr_WarnFormat(nullptr, 2, fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	inline void errorMsg(const char* fmt, Args&&... args)
	{
		// Don't modify error text if there's already a PyError set
		if ( PyErr_Occurred() )
			return;

		PyErr_SetString(PyExc_RuntimeError, formatMsg(fmt, std::forward<Args>(args)...).c_str());
	}


	// TODO: Figure out if we can do this without the const-casting?
	namespace Array
	{
		inline bool isColumnMajor(const ArrayType* im) { return (CHECK_ARRAY_FLAGS(im,NPY_ARRAY_FARRAY_RO) && !CHECK_ARRAY_FLAGS(im, NPY_ARRAY_C_CONTIGUOUS)); }
		inline bool isContiguous(const ArrayType* im) { return (CHECK_ARRAY_FLAGS(im, NPY_ARRAY_CARRAY_RO) || CHECK_ARRAY_FLAGS(im, NPY_ARRAY_FARRAY_RO)) ; }

		inline IdType getType(const ArrayType* im) { return (IdType)PyArray_TYPE(im); }
		inline std::size_t getNDims(const ArrayType* im) { return PyArray_NDIM(im); }
		inline DimType getDim(const ArrayType* im, int idim) { return PyArray_DIM(im, idim); }

		template <typename T>
		inline T* getData(const ArrayType* im) { return (T*)PyArray_DATA(const_cast<ArrayType*>(im)); }


		// Helper for dimensions order for row/column-major arrays
		inline std::vector<DimType> dimOrder(const DimInfo& info)
		{
			if ( info.columnMajor )
				return std::vector<DimType>(info.dims.begin(), info.dims.end());
			else
				return std::vector<DimType>(info.dims.rbegin(), info.dims.rend());
		}


		template <typename T> inline GuardOutArrayPtr create(const Script::DimInfo& info)
		{
			// Returns reverse-ordered dimensions in case of row-major array
			std::vector<DimType> dims = Script::Array::dimOrder(info);
			return GuardOutArrayPtr((ArrayType*)PyArray_EMPTY(((int)dims.size()), dims.data(), ID_FROM_TYPE(T), ((int)info.columnMajor)));
		}
	};

	inline bool isEmpty(const ObjectType* pyObj)
	{
		if ( PyTuple_Check(pyObj) )
			return (PyTuple_Size(const_cast<ObjectType*>(pyObj)) == 0);

		if ( PyList_Check(pyObj) )
			return (PyList_Size(const_cast<ObjectType*>(pyObj)) == 0);

		if ( PyArray_Check(pyObj) )
			return (PyArray_Size(const_cast<ObjectType*>(pyObj)) == 0);

		return (pyObj == nullptr || pyObj == Py_None);
	}

	inline bool isEmpty(const ArrayType* pyArray)
	{
		//ObjectType* pyObj = ((ObjectType*) const_cast<ArrayType*>(pyArray));
		return (PyArray_SIZE(const_cast<ArrayType*>(pyArray)) == 0);
	}


	// Minimal wrapper around script structure types
	// Structure array is implemented as a list of dictionaries for Python
	namespace Struct
	{
		inline const ObjectType* getVal(ObjectType* structPtr, std::size_t idx, const char* field)
		{
			ObjectType* dictPtr = PyList_GetItem(structPtr, idx);
			if ( !dictPtr )
				return nullptr;

			return PyDict_GetItemString(dictPtr, field);
		}

		inline const ObjectType* getVal(GuardOutObjectPtr& structPtr, std::size_t idx, const char* field)
		{
			return getVal(structPtr.get(), idx, field);
		}

		inline void setVal(GuardOutObjectPtr& structPtr, std::size_t idx, const char* field, ObjectType* val)
		{
			ObjectType* dictPtr = PyList_GetItem(structPtr.get(), idx);
			if ( !dictPtr )
				return;

			PyDict_SetItemString(dictPtr, field, val);
		}

		// Wrapper around script structure-array creation/access
		inline GuardOutObjectPtr create(std::size_t size, const std::vector<const char*>& fields)
		{
			GuardOutObjectPtr list = GuardOutObjectPtr(PyList_New(size));
			for ( std::size_t i=0; i < size; ++i )
				PyList_SetItem(list.get(), i, PyDict_New());

			return list;
		}
	};

	struct Converter : public ConvertErrors
	{
	private:
		////////////
		// Private helpers (e.g. array copy)
		struct CopyDefault
		{
			template <typename T, ENABLE_CHK(INT_MATCH(T))>
			inline static constexpr T index(T length, T idx){return idx;}
		};

		struct CopyReverse
		{
			template <typename T, ENABLE_CHK(INT_MATCH(T))>
			inline static constexpr T index(T length, T idx) { return length-idx-1; }
		};

		template <typename CopyDir = CopyDefault, typename OutT, typename InT>
		inline static void copyConvertArray(OutT* outPtr, const InT* inPtr, std::size_t length)
		{
			for ( std::size_t i = 0; i < length; ++i )
				outPtr[i] = static_cast<OutT>(inPtr[CopyDir::index(length,i)]);
		}

		template <typename T>
		static void copyConvertArray(T* outPtr, const T* inPtr, std::size_t length)
		{
			std::memcpy(outPtr, inPtr, length*sizeof(T));
		}


		template <typename CopyDir = CopyDefault, typename T>
		static void arrayCopy(T* outPtr, PyArrayObject* pyArray)
		{
			DimInfo info = getDimInfo(pyArray);
			if ( !info.contiguous )
				throw ArrayTypeError("Numpy array must be contiguous");

			std::size_t array_size = PyArray_SIZE(pyArray);
			Script::IdType type = Array::getType(pyArray);
			if ( type == ID_FROM_TYPE(bool) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<bool>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(uint8_t) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<uint8_t>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(uint16_t) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<uint16_t>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(int16_t) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<int16_t>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(uint32_t) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<uint32_t>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(int32_t) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<int32_t>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(float) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<float>(pyArray), array_size);
			else if ( type == ID_FROM_TYPE(double) )
				copyConvertArray<CopyDir>(outPtr, Array::getData<double>(pyArray), array_size);
			else
				throw ArrayTypeError("Unsupported numpy array type: %x", type);
		}


	public:
		////////////
		// Basic scalar to Python conversions
		template <typename T, ENABLE_CHK(IS_BOOL(T))>
		inline static ObjectType* fromNumeric(T val) { return PyBool_FromLong(val); }

		template <typename T, ENABLE_CHK(INT_SGN_MATCH(T, int64_t))>
		inline static ObjectType* fromNumeric(T val) { return PyLong_FromLongLong(val); }

		template <typename T, ENABLE_CHK(INT_SGN_MATCH(T, uint64_t))>
		inline static ObjectType* fromNumeric(T val) { return PyLong_FromUnsignedLongLong(val); }

		template <typename T, ENABLE_CHK(FLOAT_MATCH(T))>
		inline static ObjectType* fromNumeric(T val) { return PyFloat_FromDouble(val); }

		// Basic Python to scalar conversions
		template <typename T, ENABLE_CHK(IS_BOOL(T))>
		inline static T toNumeric(ObjectType* pyObj)
		{
			if ( PyBool_Check(pyObj) )
				return (pyObj == Py_True);

			throw ScalarConvertError("Expected a boolean argument");
		}

		template <typename T, ENABLE_CHK(INT_SGN_MATCH(T, int64_t))>
		inline static T toNumeric(ObjectType* pyObj)
		{
			if ( PyLong_Check(pyObj) )
				return static_cast<T>(PyLong_AsLongLong(pyObj));

			throw ScalarConvertError("Expected an integer argument");
		}

		template <typename T, ENABLE_CHK(INT_SGN_MATCH(T, uint64_t))>
		inline static T toNumeric(ObjectType* pyObj)
		{
			if ( PyLong_Check(pyObj) )
				return static_cast<T>(PyLong_AsUnsignedLongLong(pyObj));

			throw ScalarConvertError("Expected an integer argument");
		}

		template <typename T, ENABLE_CHK(FLOAT_MATCH(T))>
		inline static T toNumeric(ObjectType* pyObj)
		{
			if ( PyLong_Check(pyObj) )
				return static_cast<T>(PyLong_AsDouble(pyObj));
			else if ( PyFloat_Check(pyObj) )
				return static_cast<T>(PyFloat_AsDouble(pyObj));

			throw ScalarConvertError("Expected a numeric argument");
		}


		// Python string conversion
		inline static ObjectType* fromString(const char* str)
		{
			return PyStr_FromString(str);
		}

		inline static ObjectType* fromString(const std::string& str)
		{
			return fromString(str.c_str());
		}

		inline static std::string toString(ObjectType* pyObj)
		{
			if ( !PyStr_Check(pyObj) )
				throw ArgConvertError(nullptr, "Expected a string argument");

			return PyStr_AsUTF8(pyObj);
		}


		////////////
		// Vec to Python (tuple) conversion
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ObjectType* fromVec(const Vec<T>& vec)
		{
			ObjectType* outPtr = PyTuple_New(3);
			for ( int i=0; i < 3; ++i )
				PyTuple_SetItem(outPtr, i, fromNumeric(vec.e[i]));

			return outPtr;
		}

		// Python (list,tuple,array) to Vec conversions
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static Vec<T> listToVec(ObjectType* pyList)
		{
			Py_ssize_t list_size = PyList_Size(pyList);
			if ( list_size != 3 )
				throw VectorConvertError("List must have 3 numeric values");

			Vec<T> vec;
			for ( int i=0; i < list_size; ++i )
				vec.e[i] = toNumeric<T>(PyList_GetItem(pyList, list_size-i-1));

			return vec;
		}

		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static Vec<T> tupleToVec(ObjectType* pyTuple)
		{
			Py_ssize_t list_size = PyTuple_Size(pyTuple);
			if ( list_size != 3 )
				throw VectorConvertError("Tuple must have 3 numeric values");

			Vec<T> vec;
			for ( int i=0; i < list_size; ++i )
				vec.e[i] = toNumeric<T>(PyTuple_GetItem(pyTuple, list_size-i-1));

			return vec;
		}

		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static Vec<T> arrayToVec(ArrayType* pyArray)
		{
			size_t ndim = Script::Array::getNDims(pyArray);
			if ( ndim > 1 )
				throw VectorConvertError("Array must be 1-D with 3 numeric values");

			Script::DimType size = Script::Array::getDim(pyArray, 0);
			if ( size != 3 )
				throw VectorConvertError("Array must be 1-D with 3 numeric values");

			Vec<T> vec;
			arrayCopy<CopyReverse>(vec.e, pyArray);

			return vec;
		}

		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static Vec<T> toVec(ObjectType* pyObj)
		{
			try
			{
				if ( PyList_Check(pyObj) )
					return listToVec<T>(pyObj);
				else if ( PyTuple_Check(pyObj) )
					return tupleToVec<T>(pyObj);
				else if ( PyArray_Check(pyObj) )
					return arrayToVec<T>(reinterpret_cast<ArrayType*>(pyObj));
			}
			catch ( ScalarConvertError& )
			{
				throw VectorConvertError("Invalid value: Expected 3 numeric values");
			}

			throw VectorConvertError("Must be numeric list, tuple, or numpy array");
		}


		////////////
		// Python (numpy array) to ImageView conversion
		// NOTE: Unlike most other conversions this is a pointer copy not a deep copy
		//  Thus the underlying data types must match EXACTLY
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ImageView<T> toImage(ArrayType* pyArray)
		{
			if ( !PyArray_Check(pyArray) )
				throw ArrayTypeError("Expected a numpy array");

			Script::DimInfo info = Script::getDimInfo(pyArray);
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			Script::IdType type = Script::Array::getType(pyArray);
			if ( ID_FROM_TYPE(T) != type )
				throw ImageConvertError("Expected numpy array of type: %s", NAME_FROM_TYPE(T));

			return ImageView<T>(Script::Array::getData<T>(pyArray), Script::makeImageDims(info));
		}

		// Python (numpy array) to ImageOwner conversion
		// NOTE: This is a deep-copy and conversion of the array data
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ImageOwner<T> toImageCopy(ArrayType* pyArray)
		{
			if ( !PyArray_Check(pyArray) )
				throw ArrayTypeError("Expected a numpy array");

			Script::DimInfo info = Script::getDimInfo(pyArray);
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			ImageDimensions inDims = Script::makeImageDims(info);
			ImageOwner<T> outIm(inDims);

			arrayCopy(outIm.getPtr(), pyArray);
			return outIm;
		}

	};
};
