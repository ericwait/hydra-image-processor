#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

#include "HydraConfig.h"

SCR_COMMAND_CLASSDEF(CheckConfig)
{
public:
	SCR_HELP_STRING("Get Hydra library configuration information.\n"
		"Returns hydraConfig structure with configuration information.\n");


	static void execute(Script::ObjectType*& hydraConfig)
	{
		if ( !HydraConfig::validConfig() )
			Script::errorMsg("Hydra Library configuration was not initialized correctly!");

		hydraConfig = Script::Struct::create(1, {"UseProcessMutex"});
		Script::Struct::setVal(hydraConfig, 0, "UseProcessMutex", Script::Converter::fromNumeric(HydraConfig::useProcessMutex()));
	}
};
