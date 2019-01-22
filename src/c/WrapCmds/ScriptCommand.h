#include <cstddef>
#include <string>
#include <unordered_map>
#include <tuple>

#include "mph/tuple_helpers.h"

#include "PyIncludes.h"
#define DISPATCH_P_TYPE PyObject* (*) (PyObject*, PyObject*)
#define DISPATCH_FUNC											\
	static PyObject* dispatch(PyObject* self, PyObject* args)	\
	{															\
		PyObject* output = nullptr;								\
		dispatch_parsed(&output, args);							\
		return output;											\
	}


#define BEGIN_IO_TYPE_MAP(ClassName)						\
	namespace ClassName##_TypeMap							\
	{														\
		template <typename InType> using OutputType = InType;

#define END_IO_TYPE_MAP(ClassName)		\
	};

#define DEFAULT_IO_TYPE_MAP(ClassName)	\
	BEGIN_IO_TYPE_MAP(ClassName)		\
	END_IO_TYPE_MAP(ClassName)

#define SET_IO_TYPE_MAP(OutType, InType)				\
		template <> using OutputType<InType> = OutType;	\

#define GET_OUT_TYPE(ClassName, InType) ClassName##_TypeMap::OutputType<InType>

class ScriptCommand
{
public:
	using DispatchPtr = DISPATCH_P_TYPE;
	// TODO: Module initialization routines (and matlab dispatch)

private:
	static const std::string m_moduleName;
	static const std::unordered_map<std::string,DispatchPtr> m_commands;
};


template <typename Derived, typename ArgParser>
class ScriptCommandImpl : public ScriptCommand
{
public:
	using Parser = ArgParser;

	DISPATCH_FUNC;

	// Non-overloadable - Parses arguments and dispatches to execute command
	// NOTE: default execute command just
	template <typename ...ScriptArgs>
	static void dispatch_parsed(ScriptArgs&&... scriptioArgs)
	{
		// TODO: Make sure all arg-types make sense on default construction
		// NOTE: Also this requires all args are default-constructible types
		typename ArgParser::Args ioArgs;

		// TODO: try-catch for returning errors
		auto opt_args_ref = mph::tuple_subset_ref(typename ArgParser::opt_idx_seq(), ioArgs);
		opt_args_ref = Derived::defaults();

		ArgParser::parse(mph::tie_tuple(ioArgs), std::forward<ScriptArgs>(scriptioArgs)...);

		exec_dispatch(mph::tie_tuple(ioArgs));

		// Ready outputs for return
		// TODO: Convert non-deferred outputs
	}

private:
	template <typename... Args>
	static void exec_dispatch(std::tuple<Args...> ioArgs)
	{
		Derived::exec_dispatch_impl(ioArgs, mph::make_index_sequence<sizeof... (Args)>());
	}

	template <typename... Args, size_t... Is>
	static void exec_dispatch_impl(std::tuple<Args...> ioArgs, mph::index_sequence<Is...>)
	{
		Derived::execute(std::get<Is>(ioArgs)...);
	}

private:
	/////////////////////////
	// execute - (Static-overloadable)
	//   Default execute function dispatches to image-type templated
	//   process<T>(Args...) function
	/////////////////////////
	template <typename... Args>
	static void execute(Args... args)
	{
		static_assert(ArgParser::has_deferred_image_inputs(), "ArgParser has no deferred inputs. Please overload default execute() function!");

		Script::IdType type = ArgParser::getInputType(args...);

		if ( type == Script::TypeToIdMap<uint8_t>::typeId )
		{
			Derived::template process<uint8_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<uint16_t>::typeId )
		{
			Derived::template process<uint16_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<int16_t>::typeId )
		{
			Derived::template process<int16_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<uint32_t>::typeId )
		{
			Derived::template process<uint32_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<int32_t>::typeId )
		{
			Derived::template process<int32_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<float>::typeId )
		{
			Derived::template process<float>(args...);
		}
		else if ( type == Script::TypeToIdMap<double>::typeId )
		{
			Derived::template process<double>(args...);
		}
		else
		{
			// TODO: Throw type-error here
			return;
		}
	}

	// TODO: Remove this to force implementation
	template <typename T, typename... Args>
	static void process(Args... args)
	{

	}
};
