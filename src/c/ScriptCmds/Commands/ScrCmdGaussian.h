#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(Gaussian)
{
public:
	SCR_HELP_STRING("Gaussian smoothing.\n"
					"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"
					"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"
					"\t\thow to stride or jump to the next spatial block.\n"
					"\n"
					"\tSigmas = This should be an array of three positive values that represent the standard deviation of a Gaussian curve.\n"
					"\t\tZeros (0) in this array will not smooth in that direction.\n"
					"\n"
					"\tnumIterations (optional) =  This is the number of iterations to run the max filter for a given position.\n"
					"\t\tThis is useful for growing regions by the shape of the structuring element or for very large neighborhoods.\n"
					"\t\tCan be empty an array [].\n"
					"\n"
					"\tdevice (optional) = Use this if you have multiple devices and want to select one explicitly.\n"
					"\t\tSetting this to [] allows the algorithm to either pick the best device and/or will try to split\n"
					"\t\tthe data across multiple devices.\n"
					"\n"
					"\timageOut = This will be an array of the same type and shape as the input array.\n");
};
