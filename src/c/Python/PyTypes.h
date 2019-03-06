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
	inline void errorMsg(const char* fmt, Args&&... args)
	{
		// Don't modify error text if there's already a PyError set
		if ( PyErr_Occurred() )
			return;

		PyErr_SetString(PyExc_RuntimeError, formatMsg(fmt, std::forward<Args>(args)...).c_str());
	}

	inline std::vector<DimType> arrayDims(const DimInfo& info)
	{
		if ( info.columnMajor )
			return std::vector<DimType>(info.dims.begin(), info.dims.end());
		else
			return std::vector<DimType>(info.dims.rbegin(), info.dims.rend());
	}

	template <typename T> ArrayType* createArray(const Script::DimInfo& info)
	{
		// Returns reverse-ordered dimensions in case of row-major array
		std::vector<DimType> dims = Script::arrayDims(info);
		return ((ArrayType*) PyArray_EMPTY(((int)dims.size()), dims.data(), ID_FROM_TYPE(T), ((int)info.columnMajor)));
	}

	// TODO: Figure out if we can do this without the const-casting?
	namespace ArrayInfo
	{
		inline bool isColumnMajor(const ArrayType* im) { return (CHECK_ARRAY_FLAGS(im,NPY_ARRAY_FARRAY_RO) && !CHECK_ARRAY_FLAGS(im, NPY_ARRAY_C_CONTIGUOUS)); }
		inline bool isContiguous(const ArrayType* im) { return (CHECK_ARRAY_FLAGS(im, NPY_ARRAY_CARRAY_RO) || CHECK_ARRAY_FLAGS(im, NPY_ARRAY_FARRAY_RO)) ; }

		inline IdType getType(const ArrayType* im) { return (IdType)PyArray_TYPE(im); }
		inline std::size_t getNDims(const ArrayType* im) { return PyArray_NDIM(im); }
		inline DimType getDim(const ArrayType* im, int idim) { return PyArray_DIM(im, idim); }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*)PyArray_DATA(const_cast<ArrayType*>(im)); }
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

			void* data = PyArray_DATA(pyArray);
			std::size_t array_size = PyArray_SIZE(pyArray);

			Script::IdType type = ArrayInfo::getType(pyArray);
			if ( type == ID_FROM_TYPE(bool) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<bool*>(data), array_size);
			else if ( type == ID_FROM_TYPE(uint8_t) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<uint8_t*>(data), array_size);
			else if ( type == ID_FROM_TYPE(uint16_t) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<uint16_t*>(data), array_size);
			else if ( type == ID_FROM_TYPE(int16_t) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<int16_t*>(data), array_size);
			else if ( type == ID_FROM_TYPE(uint32_t) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<uint32_t*>(data), array_size);
			else if ( type == ID_FROM_TYPE(int32_t) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<int32_t*>(data), array_size);
			else if ( type == ID_FROM_TYPE(float) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<float*>(data), array_size);
			else if ( type == ID_FROM_TYPE(double) )
				copyConvertArray<CopyDir>(outPtr, reinterpret_cast<double*>(data), array_size);
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
		inline static std::string toString(ObjectType* pyObj)
		{
			if ( PyStr_Check(pyObj) )
				return PyStr_AsUTF8(pyObj);

			throw ArgConvertError(nullptr, "Expected a string argument");
		}

		inline static PyObject* fromString(const char* str)
		{
			return PyStr_FromString(str);
		}

		inline static PyObject* fromString(const std::string& str)
		{
			return fromString(str.c_str());
		}


		////////////
		// Vec to Python (tuple) conversion
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ObjectType* fromVec(const Vec<T>& vec)
		{
			PyObject* outPtr = PyTuple_New(3);
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
			size_t ndim = Script::ArrayInfo::getNDims(pyArray);
			if ( ndim > 1 )
				throw VectorConvertError("Array must be 1-D with 3 numeric values");

			Script::DimType size = Script::ArrayInfo::getDim(pyArray, 0);
			if ( size != 3 )
				throw VectorConvertError("Array must be 1-D with 3 numeric values");

			Vec<T> vec;
			arrayCopy<CopyReverse>(vec.e, pyArray);
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

			// TODO: Do we need to allow overloads for array checks?
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			Script::IdType type = Script::ArrayInfo::getType(pyArray);
			if ( Script::TypeToIdMap<T>::typeId != type )
				throw ImageConvertError("Expected numpy array of type: %s", NAME_FROM_TYPE(T));

			return ImageView<T>(Script::ArrayInfo::getData<T>(pyArray), Script::makeImageDims(info));
		}

		// Python (numpy array) to ImageOwner conversion
		// NOTE: This is a deep-copy and conversion of the array data
		template <typename T, ENABLE_CHK(NUMERIC_MATCH(T))>
		inline static ImageOwner<T> toImageCopy(ArrayType* pyArray)
		{
			if ( !PyArray_Check(pyArray) )
				throw ArrayTypeError("Expected a numpy array");

			Script::DimInfo info = Script::getDimInfo(pyArray);

			ImageDimensions inDims = Script::makeImageDims(info);
			ImageOwner<T> out(inDims);

			arrayCopy(out.getPtr(), pyArray);
			return out;
		}

	};


	bool pyobjToVec(PyObject* list_array, Vec<double>& outVec);
};
