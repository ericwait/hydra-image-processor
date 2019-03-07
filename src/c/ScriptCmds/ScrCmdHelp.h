#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(Help)
{
public:
	SCR_HELP_STRING("Print detailed usage information for the specified command.");

	static void execute(const std::string& commandName)
	{
		if ( commandName.empty() )
			ScriptCommand::printUsage();

		auto cmd = ScriptCommand::findCommand(commandName);
		if ( cmd )
			Script::writeMsg("%s\n", cmd->help().c_str());
	}
};
