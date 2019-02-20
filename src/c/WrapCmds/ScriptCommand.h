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
	struct AssertProcessFunc
	{
		template <typename... Args> static void run(Args&&...) {}

		// This condition (!is_same<T,T> is always false) forces waiting for dependent type instantiation
		static_assert(!std::is_same<T,T>::value, "Overload ::execute or ::process<OutT,InT> method in script command subclass, or define script command using DEF_SCRIPT_COMMAND_AUTO.");
	};

	using ProcessFunc = AssertProcessFunc<Derived>;
public:
	// Argument load/conversion access types
	using ArgParser = Parser;
	using ArgError = typename ArgParser::ArgError;

	// Script/converted argument types from arg parser
	using ScriptTypes = typename ArgParser::ScriptTypes;
	using ArgTypes = typename ArgParser::ArgTypes;

	// Pointers to arguments
	using ScriptPtrs = typename ArgParser::ScriptPtrs;
	using ArgPtrs = typename ArgParser::ArgPtrs;

	// Selectors
	using OptionalSel = typename ArgParser::OptionalSel;
	using DeferredSel = typename ArgParser::DeferredSel;
	using DeferredOutSel = typename ArgParser::DeferredOutSel;
	using DeferredOutImSel = typename ArgParser::DeferredOutImSel;
	using DeferredInOptSel = typename ArgParser::DeferredInOptSel;
	using NondeferredSel = typename ArgParser::NondeferredSel;
	using NondeferOutSel = typename ArgParser::NondeferOutSel;
	using NondeferInOptSel = typename ArgParser::NondeferInOptSel;

	// Deferred concrete-types
	template <typename OutT, typename InT>
	using ConcreteArgTypes = typename ArgParser::template ConcreteArgTypes<OutT, InT>;

	template <typename InT>
	using OutMap = typename ArgParser::template OutMap<InT>;

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
			// Subset of arguments that have concrete type (non-deferred)
			using ConcreteArgs = typename NondeferredSel::template type<ArgTypes>;

			// Points to the C++ arguments converted from script types
			//  NOTE: These are partially converted as the deferred arguments are converted in ::process<>
			ArgPtrs convertedPtrs;

			// Memory for script objects corresponding to arguments
			// NOTE: These local objects are necessary for temporary ownership of non-deferred Python outputs
			ScriptTypes scriptObjects;
			ScriptPtrs scriptPtrs = mph::tuple_addr_of(scriptObjects);

			// Load script pointers from script engine inputs
			ArgParser::load(scriptPtrs, std::forward<ScriptInterface>(scriptioArgs)...);

			// NOTE: Requires that args are default-constructible types
			ConcreteArgs convertedArgs;

			// Hook up convertedPtrs (non-deferred point to concreteArgs, deferred same as scriptPtrs)
			DeferredSel::select(convertedPtrs) = DeferredSel::select(scriptPtrs);
			NondeferredSel::select(convertedPtrs) = mph::tuple_addr_of(convertedArgs);

			// Load default values for optional arguments (can't be deferred)
			ArgParser::setOptionalDefaults(convertedPtrs);

			// Convert non-deferred inputs to appropriate arg types
			ArgParser::convertSelected(convertedPtrs, scriptPtrs, NondeferInOptSel{});

			// TODO: Figure out how to solve the dims inference problem
			// TODO: Currently no support for creating non-deferred output images
			// Create non-deferred outputs
			//ArgParser::createOutIm(convertedPtrs, scriptPtrs, dims, NondeferOutImSel{});

			// Run backend cuda filter using default execute() -> process<OutT,InT>()
			// or run overloaded ::execute or ::process<OutT,InT> functions
			exec_dispatch(mph::tuple_deref(convertedPtrs));

			// Convert outputs to script types
			ArgParser::convertSelected(scriptPtrs, convertedPtrs, NondeferOutSel{});

			// Load all outputs into script output structure (Necessary for Python)
			ArgParser::store(scriptPtrs, std::forward<ScriptInterface>(scriptioArgs)...);
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
		static_assert(ArgParser::has_deferred_image_inputs(), "Argument layout has no deferred inputs. Please overload default ::execute() function!");

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
		// Deferred types (Can now be fully defined)
		using DeferredArgs = ConcreteArgTypes<OutType, InType>;
		using DeferredPtrs = typename mph::tuple_ptr_t<DeferredArgs>;

		using DeferredInArgs = typename DeferredInOptSel::template type<DeferredArgs>;
		using DeferredOutArgs = typename DeferredOutSel::template type<DeferredArgs>;

		auto argRefs = std::tie(args...);
		ArgPtrs argPtrs = mph::tuple_addr_of(argRefs);

		// Pointers to fully-converted arguments to be passed (by reference) to the cuda processing function
		DeferredPtrs convertedPtrs;

		// Storage for deferred in/out types
		DeferredInArgs concreteInArgs;
		DeferredOutArgs concreteOutArgs;

		// Hook up convertedPtrs all non-deferred are already converted, deferred get hooked locals
		NondeferredSel::select(convertedPtrs) = NondeferredSel::select(argPtrs);
		DeferredInOptSel::select(convertedPtrs) = mph::tuple_addr_of(concreteInArgs);
		DeferredOutSel::select(convertedPtrs) = mph::tuple_addr_of(concreteOutArgs);

		// Convert deferred inputs to appropriate arg types
		ArgParser::convertSelected(convertedPtrs, argPtrs, DeferredInOptSel{});

		// Create imageref outputs
		// TODO: Better image dims inference
		Script::DimInfo dimInfo = ArgParser::getInputDimInfo(argRefs);
		ArgParser::createOutImRefs(convertedPtrs, argPtrs, dimInfo);

		// Run backend cuda filter
		run_dispatch(mph::tuple_deref(convertedPtrs));

		// Convert deferred outputs to script types
		ArgParser::convertSelected(argPtrs, convertedPtrs, DeferredOutSel{});
	}

	template <typename... Args>
	static void run_dispatch(std::tuple<Args...> runArgs)
	{
		Derived::run_dispatch_impl(runArgs, mph::make_index_sequence<sizeof... (Args)>());
	}

	template <typename... Args, size_t... Is>
	static void run_dispatch_impl(std::tuple<Args...> runArgs, mph::index_sequence<Is...>)
	{
		Derived::ProcessFunc::run(std::forward<Args>(std::get<Is>(runArgs))...);
	}
};
