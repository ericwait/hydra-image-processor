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
			print_all();

		auto cmd = ScriptCommand::findCommand(commandName);
		if ( cmd )
			Script::writeMsg("%s\n", cmd->help().c_str());
	}

private:
	static void print_all()
	{
		ScriptCommand::CommandList cmds = ScriptCommand::commands();
		for ( const auto& it: cmds )
			Script::writeMsg("%s\n", it.second.usage().c_str());
	}

};
