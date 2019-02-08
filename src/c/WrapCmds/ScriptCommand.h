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
		convert_dispatch(output, args);							\
		return output;											\
	}


#define SCR_DEFAULT_IO_TYPE_MAP template <typename InT> static InT get_out(InT)
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
	SCR_DEFAULT_IO_TYPE_MAP;

	// This to be an instanced struct type for compiler compatibility
	template <typename InT>
	struct OutMap_Impl
	{
		using type = decltype(Derived::get_out(std::declval<InT>()));
	};

	// Simplified type-mapping alias access alias
	template <typename InT>
	using OutMap = typename OutMap_Impl<InT>::type;

	// Argument load/conversion access types
	using ArgParser = Parser;
	using ArgError = typename ArgParser::ArgError;

	// Selectors
	using OptionalSel = typename ArgParser::OptionalSel;
	using DeferredSel = typename ArgParser::DeferredSel;
	using NondeferredSel = typename ArgParser::NondeferredSel;
	using NondeferOutSel = typename ArgParser::NondeferOutSel;
	using NondeferInOptSel = typename ArgParser::NondeferInOptSel;

	// Script engine-dependent function to dispatch parameters
	DISPATCH_FUNC;

	// Non-overloadable - Handles argument conversion and dispatches to execute command
	// NOTE: default execute command checks for deferred types and passes to
	//   the templated process function.
	template <typename... ScriptInterface>
	static void convert_dispatch(ScriptInterface&&... scriptioArgs)
	{
		try
		{
			// Script/converted argument types from arg parser
			using ScriptTypes = typename ArgParser::ScriptTypes;
			using ArgTypes = typename ArgParser::ArgTypes;

			// Pointers to arguments
			using ScriptPtrs = typename ArgParser::ScriptPtrs;
			using ArgPtrs = typename ArgParser::ArgPtrs;

			// Subset of arguments that have concrete type (non-deferred)
			using ConcreteArgs = typename NondeferredSel::template type<ArgTypes>;

			ArgPtrs argPtrs;

			// Memory for script objects corresponding to arguments
			// NOTE: These local objects are necessary for temporary ownership of non-deferred Python outputs
			ScriptTypes scriptObjects;
			ScriptPtrs scriptPtrs = mph::tuple_addr_of(scriptObjects);

			// Load script pointers from script engine inputs
			ArgParser::load(scriptPtrs, std::forward<ScriptInterface>(scriptioArgs)...);

			// NOTE: Requires that args are default-constructible types
			ConcreteArgs concreteArgs;

			// Hook up argPtrs (non-deferred point to concreteArgs, deferred same as scriptPtrs)
			DeferredSel::select(argPtrs) = DeferredSel::select(scriptPtrs);
			NondeferredSel::select(argPtrs) = mph::tuple_addr_of(concreteArgs);

			// Load default values for optional arguments (can't be deferred)
			auto optRefs = mph::tuple_deref(OptionalSel::select(argPtrs));
			optRefs = Derived::defaults();

			// Convert non-deferred inputs to appropriate arg types
			ArgParser::convertIn(argPtrs, scriptPtrs, NondeferInOptSel{});

			// TODO: Figure out how to solve the dims inference problem
			// TODO: Currently no support for creating non-deferred output images
			// Create non-deferred outputs
			//ArgParser::createOutIm(argPtrs, scriptPtrs, dims, NondeferOutImSel{});

			// Run backend cuda filter using default execute() -> process<OutT,InT>()
			// or run overloaded ::execute or ::process<OutT,InT> functions
			exec_dispatch(mph::tuple_deref(argPtrs));

			// Convert outputs to script types
			ArgParser::convertOut(scriptPtrs, argPtrs, NondeferOutSel{});

			// Load all outputs into script output structure (Necessary for Python)
			//ArgParser::store(scriptPtrs, std::forward<ScriptInterface>(scriptioArgs)...);
		}
		catch (ArgError& ae)
		{
			//TODO: Print error and usage (use Script::ErrorMsg to ignore if PyErr set)
			Derived::commandName();
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
		Derived::execute(std::forward<Args>(std::get<Is>(ioArgs))...);
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
	static void execute(Args&&... args)
	{
		static_assert(ArgParser::has_deferred_image_inputs(), "ArgParser has no deferred inputs. Please overload default execute() function!");

		Script::IdType type = ArgParser::getInputType(std::forward<Args>(args)...);

		if ( type == Script::TypeToIdMap<bool>::typeId )
		{
			Derived::template process<OutMap<bool>,bool>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<uint8_t>::typeId )
		{
			Derived::template process<OutMap<uint8_t>,uint8_t>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<uint16_t>::typeId )
		{
			Derived::template process<OutMap<uint16_t>,uint16_t>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<int16_t>::typeId )
		{
			Derived::template process<OutMap<int16_t>,int16_t>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<uint32_t>::typeId )
		{
			Derived::template process<OutMap<uint32_t>,uint32_t>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<int32_t>::typeId )
		{
			Derived::template process<OutMap<int32_t>,int32_t>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<float>::typeId )
		{
			Derived::template process<OutMap<float>,float>(std::forward<Args>(args)...);
		}
		else if ( type == Script::TypeToIdMap<double>::typeId )
		{
			Derived::template process<OutMap<double>,double>(std::forward<Args>(args)...);
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
	static void process(Args&&... args)
	{
		

		// TODO: Need to infer output dimensions
		//ProcessFunc t;
	}
};
