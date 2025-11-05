/**
 * @file ScriptCommand.h
 * @brief Base class for script command registration and dispatch
 *
 * Provides the infrastructure for registering and dispatching commands
 * that can be called from both MATLAB MEX and Python interfaces. Uses
 * a registry pattern to map command names to their implementation functions.
 */

#pragma once

#include <string>
#include <unordered_map>

#include "ScriptIncludes.h"

/// @brief Module name for the script interface
#define SCR_MODULE_NAME "HIP"

/**
 * @brief Declares a function that returns command usage string
 * @param TypeName The name of the function to declare
 */
#define SCR_USAGE_FUNC_DECL(TypeName) std::string TypeName()

/**
 * @brief Declares a function that returns command help string
 * @param TypeName The name of the function to declare
 */
#define SCR_HELP_FUNC_DECL(TypeName) std::string TypeName()

/**
 * @brief Declares a function that provides command information
 * @param TypeName The name of the function to declare
 */
#define SCR_INFO_FUNC_DECL(TypeName) void TypeName(std::string& command, std::string& help, std::string& outArgs, std::string& inArgs)

#if defined(PY_BUILD)
	#define SCR_DISPATCH_FUNC_DECL(TypeName) PyObject* TypeName(PyObject* self, PyObject* args)
	#define SCR_DISPATCH_FUNC_DEF(Name)		\
		SCR_DISPATCH_FUNC_DECL(dispatch)	\
		{									\
			PyObject* output = nullptr;		\
			convert_dispatch(output, args);	\
			return output;					\
		}
#elif defined(MEX_BUILD)
	#define SCR_DISPATCH_FUNC_DECL(TypeName) void TypeName(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
	#define SCR_DISPATCH_FUNC_DEF(Name)				\
		SCR_DISPATCH_FUNC_DECL(dispatch)			\
		{											\
			convert_dispatch(nlhs,plhs, nrhs,prhs);	\
		}
#endif

/**
 * @brief Base class for script command registration and dispatch
 *
 * Manages a registry of commands that can be called from scripting interfaces.
 * Each command has associated dispatch, usage, help, and info functions.
 * Supports both MATLAB MEX and Python interfaces through conditional compilation.
 */
class ScriptCommand
{
	/// @brief Function pointer type for command dispatch functions
	typedef SCR_DISPATCH_FUNC_DECL((*DispatchFuncType));
	/// @brief Function pointer type for usage string functions
	typedef SCR_USAGE_FUNC_DECL((*UsageFuncType));
	/// @brief Function pointer type for help string functions
	typedef SCR_HELP_FUNC_DECL((*HelpFuncType));
	/// @brief Function pointer type for command info functions
	typedef SCR_INFO_FUNC_DECL((*InfoFuncType));

public:
	/**
	 * @brief Structure holding function pointers for a command
	 *
	 * Each registered command has these four function pointers that
	 * handle dispatch, usage info, help text, and detailed information.
	 */
	struct FuncPtrs
	{
		DispatchFuncType dispatch;  ///< Main command dispatch function
		UsageFuncType usage;        ///< Returns usage string
		HelpFuncType help;          ///< Returns help text
		InfoFuncType info;          ///< Provides detailed command info
	};

	/// @brief Type for the command registry map
	using CommandList = std::unordered_map<std::string, FuncPtrs>;

	/**
	 * @brief Finds a command by name in the registry
	 *
	 * @param command The name of the command to find
	 * @return Pointer to the command's function pointers, or nullptr if not found
	 */
	inline static const FuncPtrs* findCommand(const std::string& command)
	{
		if ( m_commands.count(command) < 1 )
			return nullptr;

		return &m_commands.at(command);
	}

	/**
	 * @brief Prints usage information for all registered commands
	 */
	inline static void printUsage()
	{
		CommandList cmds = commands();
		for ( const auto& it: cmds )
			Script::writeMsg("%s\n", it.second.usage().c_str());
	}

	/**
	 * @brief Gets the map of all registered commands
	 * @return Reference to the command registry
	 */
	inline static const CommandList& commands(){return m_commands;}


protected:
	/**
	 * @brief Returns the module name for the script interface
	 * @return Module name string
	 */
	inline static const char* moduleName() {return SCR_MODULE_NAME;}

protected:
	/// @brief Static registry of all commands
	static const CommandList m_commands;
};
