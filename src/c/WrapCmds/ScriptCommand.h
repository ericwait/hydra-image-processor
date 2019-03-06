#pragma once

#include <string>
#include <unordered_map>

#include "ScriptIncludes.h"
// TODO: Put this in ifdef or py-specific includes
#define SCR_MODULE_NAME "HIP"

#define SCR_DISPATCH_FUNC_DECL(TypeName) PyObject* TypeName(PyObject* self, PyObject* args)
#define SCR_USAGE_FUNC_DECL(TypeName) std::string TypeName()
#define SCR_HELP_FUNC_DECL(TypeName) std::string TypeName()
#define SCR_INFO_FUNC_DECL(TypeName) void TypeName(std::string& command, std::string& help, std::string& outArgs, std::string& inArgs)

#define SCR_DISPATCH_FUNC_DEF(Name)		\
	SCR_DISPATCH_FUNC_DECL(dispatch)	\
	{									\
		PyObject* output = nullptr;		\
		convert_dispatch(output, args);	\
		return output;					\
	}


class ScriptCommand
{
	typedef SCR_DISPATCH_FUNC_DECL((*DispatchFuncType));
	typedef SCR_USAGE_FUNC_DECL((*UsageFuncType));
	typedef SCR_HELP_FUNC_DECL((*HelpFuncType));
	typedef SCR_INFO_FUNC_DECL((*InfoFuncType));
public:
	// TODO: Should we use virtual functions instead of static dispatch?
	struct FuncPtrs
	{
		DispatchFuncType dispatch;
		UsageFuncType usage;
		HelpFuncType help;
		InfoFuncType info;
	};

	using CommandList = std::unordered_map<std::string, FuncPtrs>;

	// TODO: Module initialization routines (and matlab dispatch)
	inline static const FuncPtrs* findCommand(const std::string& command)
	{
		if ( m_commands.count(command) < 1 )
			return nullptr;

		return &m_commands.at(command);
	}

	inline static const CommandList& commands(){return m_commands;}


protected:
	inline static const char* moduleName() {return SCR_MODULE_NAME;}

protected:
	static const CommandList m_commands;
};
