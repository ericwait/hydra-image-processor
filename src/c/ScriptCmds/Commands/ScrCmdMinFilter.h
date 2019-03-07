#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(MinFilter)
{
public:
	SCR_HELP_STRING("This will set each pixel/voxel to the max value of the neighborhood defined by the given kernel.\n"
					"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"
					"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"
					"\t\thow to stride or jump to the next spatial block.\n"
					"\n"
					"\tkernel = This is a one to three dimensional array that will be used to determine neighborhood operations.\n"
					"\t\tIn this case, the positions in the kernel that do not equal zeros will be evaluated.\n"
					"\t\tIn other words, this can be viewed as a structuring element for the max neighborhood.\n"
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
