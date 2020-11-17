#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(DeviceCount)
{
public:
	SCR_HELP_STRING("This will return the number of Cuda devices available, and their memory.\n"
					"\tNumCudaDevices -- this is the number of Cuda devices available.\n"
					"\tMemoryStats -- this is an array of structures where each entry corresponds to a Cuda device.\n"
					"\t\tThe memory structure contains the total memory on the device and the memory available for a Cuda call.\n");


	static void execute(int32_t& numDevices, Script::GuardOutObjectPtr& memStats)
	{
		std::size_t* memSizes;
		numDevices = memoryStats(&memSizes);

		memStats = Script::Struct::create(numDevices, {"total","available"});
		for ( int i=0; i < numDevices; ++i )
		{
			Script::ObjectType* total = Script::Converter::fromNumeric(memSizes[i*2]);
			Script::ObjectType* avail = Script::Converter::fromNumeric(memSizes[i*2+1]);

			Script::Struct::setVal(memStats, i, "available", avail);
			Script::Struct::setVal(memStats, i, "total", total);
		}

		delete[] memSizes;
	}
};
