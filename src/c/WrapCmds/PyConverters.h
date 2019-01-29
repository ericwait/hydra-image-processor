#include "../Cuda/ImageView.h"

#include "mph/tuple_helpers.h"
#include "mph/qualifier_helpers.h"

#include "ScriptTraits.h"

#include <tuple>
#include <string>
#include <cstdio>
#include <memory>
#include <type_traits>

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
 #define SNPRINTF snprintf
#else
 #define SNPRINTF std::snprintf
#endif

namespace Script
{	
	// strip_outer - Remove one layer of nested types (used to remove In/Out/OptParam qualifier types)
	template <typename T>
	struct strip_outer {};

	template <template<typename> class T, typename U>
	struct strip_outer<T<U>>
	{
		using type = U;
	};


	// Type traits for concrete/deferred io
	template <typename T>
	struct is_deferred
	{
		static constexpr bool value = std::is_same<T, DeferredType>::value;
	};

	template <template<typename> class T, typename U>
	struct is_deferred<T<U>>
	{
		static constexpr bool value = is_deferred<U>::value;
	};

	// Check for image types
	template <typename T>
	struct is_image
	{
		static constexpr bool value = false;
	};

	template <template<typename> class T, typename U>
	struct is_image<T<U>>
	{
		static constexpr bool value = (std::is_same<ImageRef<U>, T<U>>::value || is_image<U>::value);
	};

	// Convenience type for script inputs (e.g. const PyObject*)
	using const_script_in_t = force_const_t<Script::ObjectType*>;

	template <typename T>
	struct PyParseType
	{
		using type = const_script_in_t;
	};

	template <typename T>
	struct force_use_type
	{
		template <typename U>
		struct tfm
		{ using type = T; };
	};


	// TODO: Check for valid io-wrapping
	template <typename T> using is_outparam = std::is_same<T, OutParam<typename T::Type>>;
	template <typename T> using is_inparam = std::is_same<T, InParam<typename T::Type>>;
	template <typename T> using is_optparam = std::is_same<T, OptParam<typename T::Type>>;


	// Derivation of types for script/concrete variable tuples
	template <typename T>
	struct script_type
	{
		using type = typename T::template qualifier_t<typename T::Type::ScriptType>;
	};

	template <typename T>
	struct concrete_type
	{
		using type = typename T::template qualifier_t<typename T::Type::ConcreteType>;
	};

	template <typename Tuple>
	using script_transform = mph::tuple_type_tfm<iotrait_to_script, Tuple>;

	template <typename Tuple>
	using concrete_transform = mph::tuple_type_tfm<iotrait_to_concrete, Tuple>;

	template <typename Tuple>
	using add_ref_transform = mph::tuple_type_tfm<std::add_lvalue_reference, Tuple>;




	template <typename Derived, typename ScriptConverter, typename... Layout>
	struct ArgParser
	{
	protected:
		using ArgConvertError = typename ScriptConverter::ArgConvertError;


	public:
		// Argument type layout alias (e.g. std::tuple<OutParam<Image<Deferred>>,...>)
		using ArgLayout = std::tuple<Layout...>;

		// Script argument type layout (e.g. std::tuple<const PyArrayObject*,...>
		using ScriptTypes = typename script_transform<std::tuple<Layout...>>::type;
		using ScriptRefs = typename add_ref_transform<ScriptTypes>::type;

		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using ArgTypes = typename concrete_transform<std::tuple<Layout...>>::type;
		using ArgRefs = typename add_ref_transform<ArgTypes>::type;

		// Filter sequences for accessing different argument types
		using out_idx_seq = typename mph::filter_tuple_seq<is_outparam, ArgLayout>::type;
		using in_idx_seq = typename mph::filter_tuple_seq<is_inparam, ArgLayout>::type;
		using opt_idx_seq = typename mph::filter_tuple_seq<is_optparam, ArgLayout>::type;

		// Index sequence of input/optional args (in order of required followed by optional)
		using inopt_idx_seq = typename mph::cat_index_sequence<in_idx_seq, opt_idx_seq>::type;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using OutTypeLayout = mph::tuple_subset_t<out_idx_seq, typename mph::tuple_type_tfm<strip_outer, ArgLayout>::type>;
		using InTypeLayout = mph::tuple_subset_t<in_idx_seq, typename mph::tuple_type_tfm<strip_outer, ArgLayout>::type>;
		using OptTypeLayout = mph::tuple_subset_t<opt_idx_seq, typename mph::tuple_type_tfm<strip_outer, ArgLayout>::type>;

		// Convenience typedefs for concrete output and input argument tuples
		using OutArgs = mph::tuple_subset_t<out_idx_seq, ArgTypes>;
		using InArgs = mph::tuple_subset_t<in_idx_seq, ArgTypes>;
		using OptArgs = mph::tuple_subset_t<opt_idx_seq, ArgTypes>;


		// Sub-sequences of arguments for dealing with type-deferred inputs/outputs
		using in_im_idx_seq = typename mph::filter_tuple_subseq<is_image, ArgLayout, in_idx_seq>::type;

		using in_im_defer_idx_seq = typename mph::filter_tuple_subseq<is_deferred, ArgLayout, in_im_idx_seq>::type;
		using out_defer_idx_seq = typename mph::filter_tuple_subseq<is_deferred, ArgLayout, out_idx_seq>::type;


	public:
		static constexpr bool has_deferred_image_inputs() noexcept
		{
			return (in_im_defer_idx_seq::size() > 0);
		}

		static constexpr bool has_deferred_outputs() noexcept
		{
			return (out_defer_idx_seq::size() > 0);
		}

		template<typename... Types>
		static constexpr auto selectOptional(std::tuple<Types...>& args)
			-> mph::tuple_subset_t<opt_idx_seq, std::tuple<typename std::add_lvalue_reference<Types>::type...>>
		{
			return mph::tuple_subset_ref(opt_idx_seq(), args);
		}

		template<typename... Types>
		static constexpr auto selectInputs(std::tuple<Types...>& argsRefs)
			-> mph::tuple_subset_t<in_idx_seq, std::tuple<typename std::add_lvalue_reference<Types>::type...>>
		{
			return mph::tuple_subset_ref(in_idx_seq(), argsRefs);
		}

		template<typename... Types>
		static constexpr auto selectOutputs(std::tuple<Types...>& argsRefs)
			-> mph::tuple_subset_t<out_idx_seq, std::tuple<typename std::add_lvalue_reference<Types>::type...>>
		{
			return mph::tuple_subset_ref(out_idx_seq(), argsRefs);
		}

		template <typename... T>
		static IdType getInputType(T... ioargs)
		{
			auto in_defer_tuple = mph::tuple_subset(in_im_defer_idx_seq(), std::tuple<T...>(ioargs...));
			return Script::ArrayInfo::getType(std::get<0>(in_defer_tuple));
		}


		static void convertInputs(ArgRefs ioArgs, ScriptRefs scriptRefs)
		{
			// TODO: Potentially pre-check for conversion compatibility
			//  Converters to pass script args to actual non-deferred input types
			convert_arg_subset(ioArgs, scriptRefs, inopt_idx_seq());
		}

		template <typename InT>
		static void convertDeferredInputs(ArgRefs concreteArgs, ArgRefs ioArgs);



	public:
		// General argument error exception
		class ArgError: public std::runtime_error
		{
			static std::string make_convert_msg(const ArgConvertError& ace)
			{
				return std::string(Derived::argName(ace.getArgIndex())) + ": " + ace.what();
			}

		public:
			ArgError(const char* msg): std::runtime_error(msg)
			{}

			ArgError(const ArgConvertError& ace): std::runtime_error(make_convert_msg(ace))
			{}

		private:
			ArgError() = delete;
		};

	protected:
		template <typename... Targets, typename... Args, size_t... Is>
		static void convert_arg_subset(std::tuple<Targets...>& targets, const std::tuple<Args...>& args, mph::index_sequence<Is...>)
		{
			try
			{
				(void)std::initializer_list<int>
				{
					(ScriptConverter::convert(std::get<Is>(targets), std::get<Is>(args), Is), void(), 0)...
				};
			}
			catch (ArgConvertError& ate)
			{
				throw ArgError(ate);
			}
		}

		//template <typename... Targets, typename... Args>
		//static void convert_args(std::tuple<Targets...>& targets, const std::tuple<Args...>& args)
		//{
		//	convert_args_subset(targets, args, mph::make_index_sequence<sizeof... (Args)>());
		//}
	};





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
				: argIdx(argIdx), std::runtime_error(make_msg(fmt,args...).get())
			{}

			int getArgIndex() const { return argIdx; }
			int setArgIndex(int idx) { argIdx = idx; }

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
		template <typename T>
		static void convert(T& out, const PyObject* inPtr, std::size_t argIdx)
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

		template <typename T>
		static void convert(T& out, const PyArrayObject* inPtr, std::size_t argIdx)
		{
			convert(out, (const PyObject*)inPtr, argIdx);
		}

	private:
		// Have to check python object types when converting individual numbers
		static bool pyNumericConvert(long* out, PyObject* in)
		{
			if ( PyLong_Check(in) )
				(*out) = PyLong_AsLong(in);
			else
				return false;

			return true;
		}

		static bool pyNumericConvert(unsigned long* out, PyObject* in)
		{
			if ( PyLong_Check(in) )
				(*out) = PyLong_AsUnsignedLong(in);
			else
				return false;

			return true;
		}

		static bool pyNumericConvert(double* out, PyObject* in)
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
		static void pyArrayCopyConvert(T* outPtr, PyArrayObject* arrPtr)
		{
			// TODO: Check for contiguous
			std::size_t array_size = PyArray_SIZE(arrPtr);

			void* data = PyArray_DATA(arrPtr);
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
		static void pyArrayToImageCopy(ImageOwner<T>& out, const PyObject* inPtr)
		{
			PyArrayObject* arrPtr = const_cast<PyArrayObject*>(reinterpret_cast<const PyArrayObject*>(inPtr));
			Script::DimInfo info = Script::getDimInfo(arrPtr);

			// TODO: Do we need to allow overloads for array checks?
			if ( !info.contiguous )
				throw ImageConvertError("Only contiguous numpy arrays are supported");

			ImageDimensions inDims(Vec<std::size_t>(1),1,1);
			std::size_t nspatial = std::max<std::size_t>(3, info.dims.size());
			for ( std::size_t i=0; i < nspatial; ++i )
				inDims.dims.e[i] = info.dims[i];

			inDims.chan = (info.dims.size() >= 4) ? info.dims[3] : (1);
			inDims.frame = (info.dims.size() >= 5) ? info.dims[4] : (1);

			out = ImageOwner<T>(inDims);
			pyArrayCopyConvert(out.getPtr(), arrPtr);
		}


		// Passthrough conversion (these should already be checked for validity)
		static void convert_impl(const PyArrayObject*& outPtr, const PyObject* inPtr)
		{
			// TODO: Recheck for PyArrayObject here?
			outPtr = reinterpret_cast<const PyArrayObject*>(inPtr);
		}

		// Basic type conversions
		static void convert_impl(uint8_t& out, const PyObject* inPtr)
		{
			unsigned long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be unsigned integer value");

			out = static_cast<uint8_t>(tmp);
		}

		static void convert_impl(uint16_t& out, const PyObject* inPtr)
		{
			unsigned long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be unsigned integer value");

			out = static_cast<uint16_t>(tmp);
		}

		static void convert_impl(int16_t& out, const PyObject* inPtr)
		{
			long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Expected integer value");

			out = static_cast<int16_t>(tmp);
		}

		static void convert_impl(uint32_t& out, const PyObject* inPtr)
		{
			unsigned long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be unsigned integer value");

			out = static_cast<uint32_t>(tmp);
		}

		static void convert_impl(int32_t& out, const PyObject* inPtr)
		{
			long tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be integer value");

			out = static_cast<int32_t>(tmp);
		}

		static void convert_impl(float& out, const PyObject* inPtr)
		{
			double tmp;
			if ( !pyNumericConvert(&tmp, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be floating-point value");

			out = static_cast<float>(tmp);
		}

		static void convert_impl(double& out, const PyObject* inPtr)
		{
			if ( !pyNumericConvert(&out, const_cast<PyObject*>(inPtr)) )
				throw ScalarConvertError("Must be floating-point value");
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
			catch ( ScalarConvertError& sce )
			{
				throw VectorConvertError("Invalid value in vector, expected 3 numeric values");
			}
		}


		// Concrete ImageOwner<T> conversions
		template <typename T>
		static void convert_impl(ImageOwner<T>& out, const PyObject* inPtr)
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
		using typename BaseParser::ScriptRefs;

		// Concrete type layouts (e.g. std::tuple<PyObject*,...>)
		using typename BaseParser::ArgTypes;
		using typename BaseParser::ArgRefs;

		// IO-type stripped layout subsets (e.g. OutParam<Image<Deferred>> -> Image<Deferred>)
		using typename BaseParser::OutTypeLayout;
		using typename BaseParser::InTypeLayout;
		using typename BaseParser::OptTypeLayout;


		// Argument sequence selectors
		using typename BaseParser::out_idx_seq;
		using typename BaseParser::in_idx_seq;
		using typename BaseParser::opt_idx_seq;


		static ScriptRefs load(ScriptRefs localRefs, Script::ObjectType*& scriptOut, Script::ObjectType* scriptIn)
		{
			set_null(localRefs);
			mph::tuple_ptr_t<ScriptRefs> scriptPtrs = mph::tuple_addr_of(localRefs);

			auto inPtrs = mph::tuple_subset(in_idx_seq(), scriptPtrs);
			auto optPtrs = mph::tuple_subset(opt_idx_seq(), scriptPtrs);

			//  Make parse_string and expanded arg tuple for PyParse_Tuple
			const std::string parseStr = make_parse_str<InTypeLayout>() + "|" + make_parse_str<OptTypeLayout>();
			auto parseArgs = std::tuple_cat(expand_parse_args<InTypeLayout>(inPtrs), expand_parse_args<OptTypeLayout>(optPtrs));

			//  Call PyParse_Tuple return on error
			if ( !parse_script(scriptIn, parseStr, parseArgs) )
				throw ArgError("PyArg_ParseTuple failed");

			// TODO: Check in-params non-null
			// TODO: Check input image dimension info

			return mph::tuple_deref(scriptPtrs);
		}

	private:
		template <typename... Types, size_t... Is>
		static void set_null_impl(std::tuple<Types&...> args, mph::index_sequence<Is...>)
		{
			(void)std::initializer_list<int>
			{
				((std::get<Is>(args) = nullptr), void(), 0)...
			};
		}

		template <typename... Types>
		static void set_null(std::tuple<Types&...> args)
		{
			set_null_impl(args, mph::make_index_sequence<sizeof... (Types)>());
		}

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


		template <typename... Args, size_t... Is>
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
	};

};