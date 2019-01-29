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
		dispatch_parsed(output, args);							\
		return output;											\
	}


//#define BEGIN_IO_TYPE_MAP(ClassName)						\
//	namespace ClassName##_TypeMap							\
//	{														\
//		template <typename InType> using OutputType = InType;
//
//#define END_IO_TYPE_MAP(ClassName)		\
//	};
//
//#define DEFAULT_IO_TYPE_MAP(ClassName)	\
//	BEGIN_IO_TYPE_MAP(ClassName)		\
//	END_IO_TYPE_MAP(ClassName)
//
//#define SET_IO_TYPE_MAP(OutType, InType)				\
//		template <> using OutputType<InType> = OutType;	\
//
//#define GET_OUT_TYPE(ClassName, InType) ClassName##_TypeMap::OutputType<InType>

#define SCR_DEFAULT_IO_TYPE_MAP() template <typename InT> static InT get_out(InT)
#define SCR_DEFINE_IO_TYPE_MAP(OutT,InT) static OutT get_out(InT)


class ScriptCommand
{
public:
	using DispatchPtr = DISPATCH_P_TYPE;
	// TODO: Module initialization routines (and matlab dispatch)

protected:
	static const std::string m_moduleName;
	static const std::unordered_map<std::string,DispatchPtr> m_commands;
};


template <typename Derived, typename Parser>
class ScriptCommandImpl : public ScriptCommand
{
	template <typename T>
	struct Assert
	{
		static_assert(!std::is_same<T, T>::value, "Overload ::execute or ::process<T> method in script command subclass, or define script command using DEF_SCRIPT_COMMAND_AUTO.");
	};

	using ProcessFunc = Assert<Derived>;
public:
	// Helper types for input/output argument type mapping
	SCR_DEFAULT_IO_TYPE_MAP();

	// This to be an instanced struct type for compiler compatibility
	template <typename InT>
	struct OutMap_Impl
	{
		using type = decltype(Derived::get_out(std::declval<InT>()));
	};

	// Simplified type-mapping alias access alias
	template <typename InT>
	using OutMap = typename OutMap_Impl<InT>::type;

	using ArgParser = Parser;
	using ArgError = typename ArgParser::ArgError;

	DISPATCH_FUNC;

	// Non-overloadable - Parses arguments and dispatches to execute command
	// NOTE: default execute command checks for deferred types and passes to
	//   the templated process function.
	template <typename... ScriptInterface>
	static void dispatch_parsed(ScriptInterface&&... scriptioArgs)
	{
		try
		{
			using ScriptTypes = typename ArgParser::ScriptTypes;
			using ScriptRefs = typename ArgParser::ScriptRefs;

			using ArgTypes = typename ArgParser::ArgTypes;
			using ArgRefs = typename ArgParser::ArgRefs;

			// Memory for script objects corresponding to ioArgs
			// NOTE: These local objects are necessary for temporary ownership of non-deferred Python outputs
			ScriptTypes localObjects;
			ScriptRefs localRefs = mph::tie_tuple(localObjects);

			// NOTE: Requires all args are default-constructible types
			ArgTypes ioArgs;
			ArgRefs argRefs = mph::tie_tuple(ioArgs);

			// TODO: try-catch for returning errors
			ScriptRefs scriptRefs = ArgParser::load(localRefs, std::forward<ScriptInterface>(scriptioArgs)...);

			// Load default values for optional arguments
			auto optRefs = ArgParser::selectOptional(argRefs);
			optRefs = Derived::defaults();

			// Convert non-deferred inputs to appropriate arg types
			ArgParser::convertInputs(argRefs, scriptRefs);

			// TODO: Figure out how to solve the dims inference problem
			// TODO: Currently no support for creating non-deferred output images
			// Create non-deferred outputs
			//auto outRefs = ArgParser::selectOutputs(argRefs);
			//auto ScriptOutRefs = ArgParser::selectOutputs(scriptRefs);
			//ArgParser::createOutputs(outRefs, ScriptOutRefs);

			// Run backend cuda filter using default execute() -> process<OutT,InT>()
			// or run overloaded ::execute or ::process<OutT,InT> functions
			exec_dispatch(argRefs);

			// Convert outputs to script types
			ArgParser::convertOutputs(scriptRefs, argRefs);

			// Load all outputs into script output structure (Necessary for Python)
			ArgParser::store(scriptRefs, std::forward<ScriptInterface>(scriptioArgs)...);
		}
		catch (ArgError& ae)
		{
			//TODO: Print error and usage (use Script::ErrorMsg to ignore if PyErr set)

		}
		catch (std::exception& e)
		{
			std::string msg("Internal error: ");
			msg += e.what();
			//Script::ErrorMsg(msg.c_str());
		}
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

	static std::string usageString()
	{
		return ArgParser::outputString() + " = " + m_moduleName + "." + Derived::commandName() + ArgParser::inoptString();
	}

private:
	/////////////////////////
	// execute - (Static-overloadable)
	//   Default execute function dispatches to image-type templated
	//   process<OutT,InT>(Args...) function
	/////////////////////////
	template <typename... Args>
	static void execute(Args&... args)
	{
		static_assert(ArgParser::has_deferred_image_inputs(), "ArgParser has no deferred inputs. Please overload default execute() function!");

		Script::IdType type = ArgParser::getInputType(args...);

		if ( type == Script::TypeToIdMap<bool>::typeId )
		{
			Derived::template process<OutMap<bool>,bool>(args...);
		}
		else if ( type == Script::TypeToIdMap<uint8_t>::typeId )
		{
			Derived::template process<OutMap<uint8_t>,uint8_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<uint16_t>::typeId )
		{
			Derived::template process<OutMap<uint16_t>,uint16_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<int16_t>::typeId )
		{
			Derived::template process<OutMap<int16_t>,int16_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<uint32_t>::typeId )
		{
			Derived::template process<OutMap<uint32_t>,uint32_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<int32_t>::typeId )
		{
			Derived::template process<OutMap<int32_t>,int32_t>(args...);
		}
		else if ( type == Script::TypeToIdMap<float>::typeId )
		{
			Derived::template process<OutMap<float>,float>(args...);
		}
		else if ( type == Script::TypeToIdMap<double>::typeId )
		{
			Derived::template process<OutMap<double>,double>(args...);
		}
		else
		{
			// TODO: Throw type-error here
			return;
		}
	}

	/////////////////////////
	// process - (Static-overloadable)
	//   Default process<OutT,InT> function creates deferred io and dispatches
	//   to the class-specified cuda processing function
	/////////////////////////
	template <typename OutType, typename InType, typename... Args>
	static void process(Args&... args)
	{


		// TODO: Need to infer output dimensions
		//ProcessFunc t;
	}
};
