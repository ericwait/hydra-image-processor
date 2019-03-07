#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(GetMinMax)
{
public:
	SCR_HELP_STRING("This function finds the lowest and highest value in the array that is passed in.\n"
					"\timageIn = This is a one to five dimensional array.\n"
					"\n"
					"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"
					"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"
					"\t\tthe data across multiple devices.\n"
					"\n"
					"\tminValue = This is the lowest value found in the array.\n"
					"\tmaxValue = This is the highest value found in the array.\n");
};
