#pragma once

#include <cstddef>
#include <tuple>

#include "ScriptCommand.h"
#include "ScopedProcessMutex.h"

#include "HydraConfig.h"

#include "mph/tuple_helpers.h"

template <typename Derived, typename ConverterType>
class ScriptCommandImpl: public ScriptCommand
{
	template <typename T>
	struct AssertProcessFunc
	{
		template <typename... Args> static void run(Args&&...) {}

		// This condition (!is_same<T,T> is always false) forces waiting for dependent type instantiation
		static_assert(!std::is_same<T, T>::value, "HIP_COMPILE: Overload ::execute or ::process<OutT,InT> method in script command subclass, or define script command using SCR_CMD instead of SCR_CMD_NOPROC.");
	};

	using ProcessFunc = AssertProcessFunc<Derived>;
public:
	// Argument load/conversion access types
	using ArgConverter = ConverterType;
	using ArgError = typename ArgConverter::ArgError;

	// Script/converted argument types from arg parser
	using ScriptTypes = typename ArgConverter::ScriptTypes;
	using ArgTypes = typename ArgConverter::ArgTypes;

	// Pointers to arguments
	using ScriptPtrs = typename ArgConverter::ScriptPtrs;
	using ArgPtrs = typename ArgConverter::ArgPtrs;

	// Selectors
	using OptionalSel = typename ArgConverter::OptionalSel;
	using DeferredSel = typename ArgConverter::DeferredSel;
	using DeferredOutSel = typename ArgConverter::DeferredOutSel;
	using DeferredOutImSel = typename ArgConverter::DeferredOutImSel;
	using DeferredInOptSel = typename ArgConverter::DeferredInOptSel;
	using NondeferredSel = typename ArgConverter::NondeferredSel;
	using NondeferOutSel = typename ArgConverter::NondeferOutSel;
	using NondeferInOptSel = typename ArgConverter::NondeferInOptSel;

	// Deferred concrete-types
	template <typename OutT, typename InT>
	using ConcreteArgTypes = typename ArgConverter::template ConcreteArgTypes<OutT, InT>;

	template <typename InT>
	using OutMap = typename ArgConverter::template OutMap<InT>;

public:
	/////////
	// These are the four interface functions that are registered in the m_commands FuncPtrs list
	// (dispatch, usage, help, info)

	// Script engine-dependent function to dispatch parameters
	inline static SCR_DISPATCH_FUNC_DEF(dispatch)

		inline static SCR_HELP_FUNC_DECL(help)
	{
		if ( Derived::helpStr.size() > 0 )
			return Derived::usage() + "\n\n" + std::string(Derived::helpStr);
		else
			return Derived::usage();
	}

	inline static SCR_USAGE_FUNC_DECL(usage)
	{
		// TODO: Fix the output wrapping here
		return usageOutput(ArgConverter::outargstr())
			+ moduleName() + "." + Derived::commandName()
			+ "("+ ArgConverter::inoptargstr() + ")";
	}

	inline static SCR_INFO_FUNC_DECL(info)
	{
		command = Derived::commandName();
		help = Derived::help();
		outArgs = ArgConverter::outargstr();
		inArgs = ArgConverter::inoptargstr();
	}

private:
	inline static std::string usageOutput(const std::string& outStr)
	{
		if ( outStr.empty() )
			return outStr;
		else
			return std::string("[") + outStr + "] = ";
	}

public:

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

			// TODO: Specialize output types for python to make sure memory is cleaned up
			// Points to the C++ arguments converted from script types
			//  NOTE: These are partially converted as the deferred arguments are converted in ::process<>
			ArgPtrs convertedPtrs;

			// Memory for script objects corresponding to arguments
			// NOTE: These local objects are necessary for temporary ownership of non-deferred Python outputs
			ScriptTypes scriptObjects;
			ScriptPtrs scriptPtrs = mph::tuple_addr_of(scriptObjects);

			// Load script pointers from script engine inputs
			ArgConverter::load(scriptPtrs, std::forward<ScriptInterface>(scriptioArgs)...);

			// NOTE: Requires that args are default-constructible types
			ConcreteArgs convertedArgs;

			// Hook up convertedPtrs (non-deferred point to concreteArgs, deferred same as scriptPtrs)
			DeferredSel::select(convertedPtrs) = DeferredSel::select(scriptPtrs);
			NondeferredSel::select(convertedPtrs) = mph::tuple_addr_of(convertedArgs);

			// Load default values for optional arguments (can't be deferred)
			ArgConverter::setOptionalDefaults(convertedPtrs);

			// Convert non-deferred inputs to appropriate arg types
			ArgConverter::convertSelected(convertedPtrs, scriptPtrs, NondeferInOptSel{});

			// TODO: Figure out how to solve the dims inference problem
			// TODO: Currently no support for creating non-deferred output images
			// Create non-deferred outputs
			//ArgConverter::createOutIm(convertedPtrs, scriptPtrs, dims, NondeferOutImSel{});

			// Run backend cuda filter using default execute() -> process<OutT,InT>()
			// or run overloaded ::execute or ::process<OutT,InT> functions
			exec_dispatch(mph::tuple_deref(convertedPtrs));

			// Convert outputs to script types
			ArgConverter::convertSelected(scriptPtrs, convertedPtrs, NondeferOutSel{});

			// Load all outputs into script output structure (Necessary for Python)
			ArgConverter::store(scriptPtrs, std::forward<ScriptInterface>(scriptioArgs)...);
		}
		catch ( ArgError& ae )
		{
			Script::errorMsg(ae.what());
		}
		catch ( std::exception& e )
		{
			std::string msg("Internal error: ");
			msg += e.what();

			Script::errorMsg(msg.c_str());
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
		// TODO: MRW - Should all execute() routines dispatch through SCOPED_PROCESS_MUTEX since it can be disabled now?
		Derived::execute(std::forward<Args>(std::get<Is>(ioArgs))...);
	}

private:
	/////////////////////////
	// execute - (Static-overloadable)
	//   Default execute function dispatches to image-type templated
	//   process<OutT,InT>(Args...) function
	//
	//   NOTE: Main purpose of default execute() is to check config and
	//     dispatch to exec_internal()
	/////////////////////////
	template <typename... Args>
	static void execute(Args&&... args)
	{
		if ( HydraConfig::useProcessMutex() )
		{
			// Use a scoped process-level mutex to run only a single GPU kernel at a time
			// TODO: Figure out a scheduling system for multi-process HIP calls
			SCOPED_PROCESS_MUTEX(hip_cmd_gpu_);
			Derived::exec_internal(std::forward<Args>(args)...);
		}
		else
		{
			Derived::exec_internal(std::forward<Args>(args)...);
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
		ArgConverter::convertSelected(convertedPtrs, argPtrs, DeferredInOptSel{});

		// Create imageref outputs
		// TODO: Better image dims inference
		Script::DimInfo dimInfo = ArgConverter::getInputDimInfo(argRefs);
		ArgConverter::createOutImRefs(convertedPtrs, argPtrs, dimInfo);

		// Run backend cuda filter
		run_dispatch(mph::tuple_deref(convertedPtrs));

		// Convert deferred outputs to script types
		ArgConverter::convertSelected(argPtrs, convertedPtrs, DeferredOutSel{});
	}

	/////////////////////////
	// exec_internal - (Not overloadable)
	//   Dispatch to processing function:
	//   process<OutT,inT>()
	/////////////////////////
	template <typename... Args>
	static void exec_internal(Args&&... args)
	{
		static_assert(ArgConverter::has_deferred_image_inputs(), "HIP_COMPILE: Argument layout has no dynamic image inputs. Please overload default ::execute() function!");

		Script::IdType type = ArgConverter::getInputType(std::forward<Args>(args)...);

		if (type == Script::TypeToIdMap<bool>::typeId)
		{
			Derived::template process<OutMap<bool>, bool>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<uint8_t>::typeId)
		{
			Derived::template process<OutMap<uint8_t>, uint8_t>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<uint16_t>::typeId)
		{
			Derived::template process<OutMap<uint16_t>, uint16_t>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<int16_t>::typeId)
		{
			Derived::template process<OutMap<int16_t>, int16_t>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<uint32_t>::typeId)
		{
			Derived::template process<OutMap<uint32_t>, uint32_t>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<int32_t>::typeId)
		{
			Derived::template process<OutMap<int32_t>, int32_t>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<float>::typeId)
		{
			Derived::template process<OutMap<float>, float>(std::forward<Args>(args)...);
		}
		else if (type == Script::TypeToIdMap<double>::typeId)
		{
			Derived::template process<OutMap<double>, double>(std::forward<Args>(args)...);
		}
		else
		{
			throw ArgError("Image type unsupported (%x)", type);
		}
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
