#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(WienerFilter)
{
public:
	SCR_HELP_STRING("A Wiener filter aims to denoise an image in a linear fashion.\n"
					"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"
					"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"
					"\t\thow to stride or jump to the next spatial block.\n"
					"\n"
					"\tkernel (optional) = This is a one to three dimensional array that will be used to determine neighborhood operations.\n"
					"\t\tIn this case, the positions in the kernel that do not equal zeros will be evaluated.\n"
					"\t\tIn other words, this can be viewed as a structuring element for the neighborhood.\n"
					"\t\t This can be an empty array [] and which will use a 3x3x3 neighborhood (or equivalent given input dimension).\n"
					"\n"
					"\tnoiseVariance (optional) =  This is the expected variance of the noise.\n"
					"\t\tThis should be a scalar value or an empty array [].\n"
					"\n"
					"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"
					"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"
					"\t\tthe data across multiple devices.\n"
					"\n"
					"\timageOut = This will be an array of the same type and shape as the input array.\n");
};
