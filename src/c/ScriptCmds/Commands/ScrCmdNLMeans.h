#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(NLMeans)
{
public:
	SCR_HELP_STRING("Apply an approximate non-local means filter using patch mean and covariance with Fisher discrminant distance\n"
		"\timageIn = This is a one to five dimensional array. The first three dimensions are treated as spatial.\n"
		"\t\tThe spatial dimensions will have the kernel applied. The last two dimensions will determine\n"
		"\t\thow to stride or jump to the next spatial block.\n"
		"\n"
		"\th = weighting applied to patch difference function. typically e.g. 0.05-0.1. controls the amount of smoothing.\n"
		"\n"
		"\tsearchWindowRadius = radius of region to locate patches at.\n"
		"\n"
		"\tnhoodRadius = radius of patch size (comparison window).\n");
};
