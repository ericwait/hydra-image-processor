#include "../Cuda/ImageView.h"

#include "mph/tuple_helpers.h"
#include "mph/qualifier_helpers.h"

#include "ScriptTraits.h"
#include "ScriptTraitTfms.h"
#include "ArgParser.h"

#include <tuple>
#include <string>
#include <cstdio>
#include <memory>
#include <type_traits>

#undef snprintf

#define SCR_PARAMS(...) __VA_ARGS__

#define SCR_OUTPUT(TypeMacro) Script::OutParam<TypeMacro>
#define SCR_INPUT(TypeMacro) Script::InParam<TypeMacro>
#define SCR_OPTIONAL(TypeMacro, DefVal) Script::OptParam<TypeMacro>

#define SCR_IMAGE_CONVERT(VarName, VarType) Script::Image<VarType>
//#define SCR_IMAGE_REQUIRE(VarName, VarType) Script::ImageRef<VarType>
#define SCR_IMAGE_DYNAMIC(VarName) Script::ImageRef<Script::DeferredType>

#define SCR_SCALAR(VarName, VarType) Script::Scalar<VarType>
#define SCR_SCALAR_DYNAMIC(VarName) Script::Scalar<Script::DeferredType>

#define SCR_VECTOR(VarName, VarType) Script::Vector<VarType>
#define SCR_VECTOR_DYNAMIC(VarName) Script::Vector<Script::DeferredType>

#ifdef _WIN32
 #define SNPRINTF std::snprintf
#else
 #define SNPRINTF std::snprintf
#endif

namespace Script
{
	// Helper for python arg expanders
	template <typename T>
	struct ParserArg {};

	template <template<typename> class T, typename C>
	struct ParserArg<T<C>>
	{
		using ArgTuple = std::tuple<Script::ObjectType const**>;
		static ArgTuple argTuple(Script::ObjectType const** argPtr)
		{
			return std::make_tuple(argPtr);
		}

		// TODO: Move to compile-time strings (maybe library)
		static const char* argString()
		{
			static const char parse_type[] = "O";
			return parse_type;
		};
	};

	template <typename C>
	struct ParserArg<Image<C>>
	{
		using ArgTuple = std::tuple<PyTypeObject*, Script::ArrayType const**>;
		static ArgTuple argTuple(Script::ArrayType const** argPtr)
		{
			return std::make_tuple(&PyArray_Type, argPtr);
		}

		static const char* argString()
		{
			static const char parse_type[] = "O!";
			return parse_type;
		};
	};

	template <typename C>
	struct ParserArg<ImageRef<C>>
	{
		using ArgTuple = std::tuple<PyTypeObject*, Script::ArrayType const**>;
		static ArgTuple argTuple(Script::ArrayType const** argPtr)
		{
			return std::make_tuple(&PyArray_Type, argPtr);
		}

		static const char* argString()
		{
			static const char parse_type[] = "O!";
			return parse_type;
		};
	};



	template <typename T> struct PyToTypeMap {};
	template <> struct PyToTypeMap<uint8_t> { using type = unsigned long; };
	template <> struct PyToTypeMap<uint16_t> { using type = unsigned long; };
	template <> struct PyToTypeMap<uint32_t> { using type = unsigned long; };
	template <> struct PyToTypeMap<int16_t> { using type = unsigned long; };
	template <> struct PyToTypeMap<int32_t> { using type = unsigned long; };
	template <> struct PyToTypeMap<float> { using type = unsigned long; };
	template <> struct PyToTypeMap<double> { using type = unsigned long; };


	// Script-to-concrete input converter
	struct PyTypeConverter
	{
		class ArgConvertError: public std::runtime_error
		{
			template <typename... Args>
			static std::unique_ptr<char[]> make_msg(const char* fmt, Args... args)
			{
				size_t size = SNPRINTF(nullptr, 0, fmt, args...);

				std::unique_ptr<char[]> msgPtr(new char[size+1]);
				SNPRINTF(msgPtr.get(), size, fmt, args...);
				return msgPtr;
			}

			static std::unique_ptr<char[]> make_msg(const char* fmt)
			{
				size_t size = std::strlen(fmt);

				std::unique_ptr<char[]> msgPtr(new char[size+1]);
				std::strncpy(msgPtr.get(), fmt, size);
				return msgPtr;
			}

		public:
			template <typename... Args>
			ArgConvertError(int argIdx, const char* fmt, Args... args)
				: std::runtime_error(make_msg(fmt,args...).get()), argIdx(argIdx)
			{}

			int getArgIndex() const { return argIdx; }
			void setArgIndex(int idx) { argIdx = idx; }

		private:
			ArgConvertError() = delete;

		private:
			int argIdx;
		};

	protected:
		class ScalarConvertError: public ArgConvertError
		{
		public:
			template <typename... Args>
			ScalarConvertError(const char* fmt, Args... args)
				: ArgConvertError(-1, fmt, args...){}
		};

		class VectorConvertError: public ArgConvertError
		{
		public:
			template <typename... Args>
			VectorConvertError(const char* fmt, Args... args)
				: ArgConvertError(-1, fmt, args...) {}
		};

		class ImageConvertError: public ArgConvertError
		{
		public:
			template <typename... Args>
			ImageConvertError(const char* fmt, Args... args)
				: ArgConvertError(-1, fmt, args...) {}
		};

		class ArrayTypeError: public ArgConvertError
		{
		public:
			template <typename... Args>
			ArrayTypeError(const char* fmt, Args... args)
				: ArgConvertError(-1, fmt, args...) {}
		};

	public:
		template <typename OutT, typename InT>
		static void convertArg(OutT& out, const InT* inPtr, int argIdx)
		{
			// NOTE: if inPtr is nullptr then this is presumed to be optional
			if ( inPtr == nullptr )
				return;

			try
			{
				convert_impl(out, inPtr);
			}
			catch ( ArgConvertError& ace )
			{
				ace.setArgIndex(argIdx);
				throw;
			}
		}

		template <typename OutT, typename InT>
		static void convertArg(OutT*& outPtr, const InT& in, int argIdx)
		{
			try
			{
				convert_impl(outPtr, in);
			}
			catch ( ArgConvertError& ace )
			{
				ace.setArgIndex(argIdx);
				throw;
			}
		}

	private:
		// Have to check python object types when converting individual numbers
		static bool pyToNumericConvert(long* out, PyObject* in)
		{
			if ( PyLong_Check(in) )
				(*out) = PyLong_AsLong(in);
			else
				return false;

			return true;
		}

		static bool pyToNumericConvert(unsigned long* out, PyObject* in)
		{
			if ( PyLong_Check(in) )
				(*out) = PyLong_AsUnsignedLong(in);
			else
				return false;

			return true;
		}

		static bool pyToNumericConvert(double* out, PyObject* in)
		{
			if ( PyLong_Check(in) )
				(*out) = PyLong_AsDouble(in);
			else if ( PyFloat_Check(in) )
				(*out) = PyFloat_AsDouble(in);
			else
				return false;

			return true;
		}


		// Copy-converter for python arrays
		// NOTE: This doesn't account for stride differences and should only be used for
		//   1-D numpy arrays
		template <typename OutType, typename InType>
		static void convertArray(OutType* outPtr, const InType* inPtr, std::size_t length)
		{
			for ( std::size_t i = 0; i < length; ++i )
				outPtr[i] = static_cast<OutType>(inPtr[i]);
		}

		// Specialization runs a simple memcpy
		template <typename T>
		static void convertArray(T* outPtr, const T* inPtr, std::size_t length)
		{
			std::memcpy(outPtr, inPtr, length*sizeof(T));
		}


		template <typename T>
		static void pyArrayCopyConvert(T* outPtr, const PyArrayObject* arrPtr)
		{
			// TODO: Check for contiguous
			std::size_t array_size = PyArray_SIZE(const_cast<PyArrayObject*>(arrPtr));

			void* data = PyArray_DATA(const_cast<PyArrayObject*>(arrPtr));
			Script::IdType type = Script::ArrayInfo::getType(arrPtr);
			if ( type == NPY_BOOL )
				convertArray(outPtr, reinterpret_cast<bool*>(data), array_size);
			else if ( type == NPY_UINT8 )
				convertArray(outPtr, reinterpret_cast<uint8_t*>(data), array_size);
			else if ( type == NPY_UINT16 )
				convertArray(outPtr, reinterpret_cast<uint16_t*>(data), array_size);
			else if ( type == NPY_INT16 )
				convertArray(outPtr, reinterpret_cast<int16_t*>(data), array_size);
			else if ( type == NPY_UINT32 )
				convertArray(outPtr, reinterpret_cast<uint32_t*>(data), array_size);
			else if ( type == NPY_INT32 )
				convertArray(outPtr, reinterpret_cast<int32_t*>(data), array_size);
			else if ( type == NPY_FLOAT )
				convertArray(outPtr, reinterpret_cast<float*>(data), array_size);
			else if ( type == NPY_DOUBLE )
				convertArray(outPtr, reinterpret_cast<double*>(data), array_size);
			else
				throw ArrayTypeError("Unsupported numpy array type: %x", type);
		}


		// Helpers for Vec<T> conversion
		template <typename T>
		static void pyListToVec(Vec<T>& out, const PyObject* inPtr)
		{
			PyObject* list = const_cast<PyObject*>(inPtr);
			Py_ssize_t list_size = PyList_Size(list);
			if ( list_size != 3 )
				throw VectorConvertError("List must have 3 numeric values");

			for ( int i=0; i < list_size; ++i )
			{
				PyObject* item = PyList_GetItem(list, i);
				convert_impl(out.e[i], item);
			}
		}

		template <typename T>
		static void pyTupleToVec(Vec<T>& out, const PyObject* inPtr)
		{
			PyObject* tuple = const_cast<PyObject*>(inPtr);
			Py_ssize_t tuple_size = PyTuple_Size(tuple);
			if ( tuple_size != 3 )
				throw VectorConvertError("Tuple must have 3 numeric values");

			for ( int i=0; i < tuple_size; ++i )
			{
				PyObject* item = PyTuple_GetItem(tuple, i);
				convert_impl(out.e[i], item);
			}
		}

		template <typename T>
		static void pyArrayToVec(Vec<T>& out, const PyObject* inPtr)
		{
			PyArrayObject* arrPtr = const_cast<PyArrayObject*>(reinterpret_cast<const PyArrayObject*>(inPtr));
			size_t ndim = Script::ArrayInfo::getNDims(arrPtr);
			if ( ndim > 1 )
				throw VectorConvertError("Array must be 1-D with 3 numeric values");

			Script::DimType size = Script::ArrayInfo::getDim(arrPtr, 0);
			if ( size != 3 )
				throw VectorConvertError("Array must be 1-D with 3 numeric values");

			pyArrayCopyConvert(out.e, arrPtr);
		}


		template <typename T>
		static void pyArrayToImageCopy(ImageOwner<T>& out, const PyArrayObject* inPtr)
		{
			Script::DimInfo info = Script::getDimInfo(inPtr);

			// TODO: Do we need to allow overloads for array checks?
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			ImageDimensions inDims = Script::makeImageDims(info);
			out = ImageOwner<T>(inDims);
			pyArrayCopyConvert(out.getPtr(), inPtr);
		}

		template <typename T>
		static void pyArrayToImageRef(ImageView<T>& out, const PyArrayObject* inPtr)
		{
			Script::DimInfo info = Script::getDimInfo(inPtr);

			// TODO: Do we need to allow overloads for array checks?
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			Script::IdType type = Script::ArrayInfo::getType(inPtr);
			if ( Script::TypeToIdMap<T>::typeId != type )
				throw ImageConvertError("Expected numpy array of type: %s", Script::TypeNameMap<T>::name);

			ImageDimensions inDims = Script::makeImageDims(info);
			out = ImageView<T>(Script::ArrayInfo::getData<T>(inPtr), inDims);
		}


		//////////////////////////////////
		// Basic type conversions
		template <typename T>
		static void convert_impl(T& out, const PyObject* inPtr)
		{
			typename PyToTypeMap<T>::type tmp;
			if ( !pyToNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be unsigned integer value");

			out = static_cast<T>(tmp);
		}

		template <typename T>
		static void convert_impl(PyObject*& outPtr, const T& in)
		{
			outPtr = Script::fromNumeric(in);
		}

		// Vector conversions
		template <typename T>
		static void convert_impl(Vec<T>& out, const PyObject* inPtr)
		{
			try
			{
				if ( PyList_Check(inPtr) )
					pyListToVec(out, inPtr);
				else if ( PyTuple_Check(inPtr) )
					pyTupleToVec(out, inPtr);
				else if ( PyArray_Check(inPtr) )
					pyArrayToVec(out, inPtr);
				else
					throw VectorConvertError("Must be numeric list, tuple, or numpy array");
			}
			catch ( ScalarConvertError& )
			{
				throw VectorConvertError("Invalid value in vector, expected 3 numeric values");
			}
		}

		template <typename T>
		static void convert_impl(PyObject*& outPtr, const Vec<T>& in)
		{
			outPtr = PyTuple_New(3);
			for ( int i=0; i < 3; ++i )
				PyTuple_SetItem(outPtr, i, Script::fromNumeric(in.e[i]));
		}


		// Concrete ImageOwner<T> conversions
		template <typename T>
		static void convert_impl(ImageOwner<T>& out, const PyArrayObject* inPtr)
		{
			if ( PyArray_Check(inPtr) )
			{
				pyArrayToImageCopy(out, inPtr);
			}
			else
			{
				throw ImageConvertError("Must be a numpy array");
			}
		}


		template <typename T>
		static void convert_impl(ImageView<T>& out, const PyArrayObject* inPtr)
		{
			if ( PyArray_Check(inPtr) )
			{
				pyArrayToImageRef(out, inPtr);
			}
			else
			{
				throw ImageConvertError("Must be a numpy array");
			}
		}

		template <typename T>
		static void convert_impl(PyArrayObject*& out, const ImageView<T>& in)
		{
			if ( out == nullptr )
				throw ImageConvertError("Output image data should already be created");
		}
	};


	template <typename Derived, typename... Layout>
	struct PyArgParser : public ArgParser<Derived, PyTypeConverter, Layout...>
	{
		using BaseParser = ArgParser<Derived, PyTypeConverter, Layout...>;

		using typename BaseParser::ArgError;

		// Argument type layout alias (e.g. std::tuple<OutParam<Image<Deferred>>,...>)
		using typename BaseParser::ArgLayout;

		// Script argument type layout (e.g. std::tuple<const PyArrayObject*,...>
		using typename BaseParser::ScriptTypes;
		using typename BaseParser::ScriptPtrs;

		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using typename BaseParser::ArgTypes;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using typename BaseParser::OutTypeLayout;
		using typename BaseParser::InTypeLayout;
		using typename BaseParser::OptTypeLayout;


		static void load(ScriptPtrs& scriptPtrs, Script::ObjectType*& scriptOut, Script::ObjectType* scriptIn)
		{
			mph::tuple_fill_value(mph::tuple_deref(scriptPtrs), nullptr);

			auto inPtrs = BaseParser::InputSel::select(scriptPtrs);
			auto optPtrs = BaseParser::OptionalSel::select(scriptPtrs);

			//  Make parse_string and expanded arg tuple for PyParse_Tuple
			const std::string parseStr = make_parse_str<InTypeLayout>() + "|" + make_parse_str<OptTypeLayout>();
			auto parseArgs = std::tuple_cat(expand_parse_args<InTypeLayout>(inPtrs), expand_parse_args<OptTypeLayout>(optPtrs));

			//  Call PyParse_Tuple return on error
			if ( !parse_script(scriptIn, parseStr, parseArgs) )
				throw ArgError("PyArg_ParseTuple failed");

			// TODO: Check in-params non-null
			// TODO: Check input image dimension info
		}


		static void store(ScriptPtrs& scriptPtrs, Script::ObjectType*& scriptOut, Script::ObjectType* scriptIn)
		{
			using OutInfo = typename mph::tuple_info<typename BaseParser::OutputSel::template type<ScriptPtrs>>;

			auto outPtrs = BaseParser::OutputSel::select(scriptPtrs);
			auto outRefs = mph::tuple_deref(outPtrs);

			if ( OutInfo::size == 1 )
				scriptOut = reinterpret_cast<Script::ObjectType*>(std::get<0>(outRefs));
			else
			{
				scriptOut = PyTuple_New(OutInfo::size);
				set_out_items(scriptOut, BaseParser::OutputSel::select(scriptPtrs));
			}
		}

	private:
		template <typename... TypeLayout, typename... Args, size_t... Is>
		static constexpr auto expand_parse_args_impl(std::tuple<TypeLayout...>, const std::tuple<Args...>& args, mph::index_sequence<Is...>) noexcept
			-> decltype(std::tuple_cat(ParserArg<TypeLayout>::argTuple(std::declval<Args>())...))
		{
			return std::tuple_cat(ParserArg<TypeLayout>::argTuple(std::get<Is>(args))...);
		}

		//////
		// expand_parse_args - Transform PyParser args so they can be passed to PyArg_ParseTuple
		template <typename TypeLayout, typename... Args>
		static constexpr auto expand_parse_args(const std::tuple<Args...>& args) noexcept
			-> decltype(expand_parse_args_impl(std::declval<TypeLayout>(), std::declval<std::tuple<Args...>>(), std::declval<mph::make_index_sequence<sizeof... (Args)>>()))
		{
			return expand_parse_args_impl(TypeLayout(), args, mph::make_index_sequence<sizeof... (Args)>());
		}
		

		// TODO: Change all these to compile-time string classes
		static std::string strcat_initializer(const std::initializer_list<const char*>& strsIn)
		{
			std::string out;
			out.reserve(2*strsIn.size() + 1);
			for ( const auto& it: strsIn )
				out += it;

			return out;
		}

		template <typename... TypeLayout>
		static std::string make_parse_str_impl(std::tuple<TypeLayout...>)
		{
			return strcat_initializer({ ParserArg<TypeLayout>::argString()... });
		}

		//////
		// make_parse_str - Create a PyArgs_ParseTuple format string from layout args
		template <typename TypeLayout>
		static std::string make_parse_str()
		{
			return make_parse_str_impl(TypeLayout());
		}


		template <typename... Args, std::size_t... Is>
		static bool parse_script_impl(PyObject* scriptIn, const std::string& format, const std::tuple<Args...>& args, mph::index_sequence<Is...>)
		{
			return (PyArg_ParseTuple(scriptIn, format.c_str(), std::get<Is>(args)...) != 0);
		}

		//////
		// parse_script - Unpack inopt argument reference tuple and pass along to PyArg_ParseTuple
		template <typename... Args>
		static bool parse_script(PyObject* scriptIn, const std::string& format, const std::tuple<Args...>& argpack)
		{
			return parse_script_impl(scriptIn, format, argpack, mph::make_index_sequence<sizeof... (Args)>());
		}


		template <typename... Args, std::size_t... Is>
		static void set_out_items_impl(PyObject*& scriptTuple, const std::tuple<Args...>& argpack, mph::index_sequence<Is...>)
		{
			(void)std::initializer_list<int>
			{
				(PyTuple_SetItem(scriptTuple, Is, reinterpret_cast<Script::ObjectType*>(std::get<Is>(argpack))), void(), 0)...
			};
		}

		template <typename... Args>
		static void set_out_items(PyObject*& scriptTuple, const std::tuple<Args...>& argpack)
		{
			set_out_items_impl(scriptTuple, argpack, mph::make_index_sequence<sizeof... (Args)>{});
		}
	};

};