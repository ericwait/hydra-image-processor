#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(Info)
{
public:
	SCR_HELP_STRING("Get information on all available mex commands.\n"
					"Returns commandInfo structure array containing information on all mex commands.\n"
					"   commandInfo.command - Command string\n"
					"   commandInfo.outArgs - Comma-delimited string list of output arguments\n"
					"   commandInfo.inArgs - Comma-delimited string list of input arguments\n"
					"   commandInfo.helpLines - Help string\n");


	static void execute(Script::ObjectType*& cmdInfo)
	{
		ScriptCommand::CommandList cmds = ScriptCommand::commands();

		cmdInfo = Script::Struct::create(cmds.size(), {"command", "outArgs", "inArgs", "help"});

		ScriptCommand::CommandList::const_iterator it = cmds.cbegin();
		for ( int i=0; it != cmds.cend(); ++it, ++i )
		{
			std::string command;
			std::string outArgs;
			std::string inArgs;
			std::string help;

			const ScriptCommand::FuncPtrs& cmdFuncs = it->second;
			cmdFuncs.info(command, help, outArgs, inArgs);

			Script::Struct::setVal(cmdInfo, i, "command", Script::Converter::fromString(command));
			Script::Struct::setVal(cmdInfo, i, "outArgs", Script::Converter::fromString(outArgs));
			Script::Struct::setVal(cmdInfo, i, "inArgs", Script::Converter::fromString(inArgs));
			Script::Struct::setVal(cmdInfo, i, "help", Script::Converter::fromString(help));
		}
	}
};
