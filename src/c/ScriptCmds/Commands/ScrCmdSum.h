#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(Sum)
{
public:
	SCR_HELP_STRING("This sums up the entire array in.\n"
					"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"
					"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"
					"\t\thow to stride or jump to the next spatial block.\n"
					"\n"
					"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"
					"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"
					"\t\tthe data across multiple devices.\n"
					"\n"
					"\tvalueOut = This is the summation of the entire array.\n");
};
