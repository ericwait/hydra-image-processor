#pragma once

#include "ScriptCommandImpl.h"
#include "ScriptCommandDefines.h"

SCR_COMMAND_CLASSDEF(DeviceStats)
{
public:
	SCR_HELP_STRING("This will return the statistics of each Cuda capable device installed.\n"
					"\tDeviceStatsArray -- this is an array of structs, one struct per device.\n"
					"\t\tThe struct has these fields: name, major, minor, constMem, sharedMem, totalMem, tccDriver, mpCount, threadsPerMP, warpSize, maxThreads.");


	static void execute(Script::GuardOutObjectPtr& deviceStruct)
	{
		DevStats* devStats;
		int numDevices = deviceStats(&devStats);

		deviceStruct = Script::Struct::create(numDevices, 
			{ "name", "major", "minor", "constMem", "sharedMem", 
			"totalMem", "tccDriver", "mpCount", "threadsPerMP", 
			"warpSize", "maxThreads" });

		for ( int i=0; i < numDevices; ++i )
		{
			Script::Struct::setVal(deviceStruct, i, "name", Script::Converter::fromString(devStats[i].name));
			Script::Struct::setVal(deviceStruct, i, "major", Script::Converter::fromNumeric(devStats[i].major));
			Script::Struct::setVal(deviceStruct, i, "minor", Script::Converter::fromNumeric(devStats[i].minor));
			Script::Struct::setVal(deviceStruct, i, "constMem", Script::Converter::fromNumeric(devStats[i].constMem));
			Script::Struct::setVal(deviceStruct, i, "sharedMem", Script::Converter::fromNumeric(devStats[i].sharedMem));
			Script::Struct::setVal(deviceStruct, i, "totalMem", Script::Converter::fromNumeric(devStats[i].totalMem));
			Script::Struct::setVal(deviceStruct, i, "tccDriver", Script::Converter::fromNumeric(devStats[i].tccDriver));
			Script::Struct::setVal(deviceStruct, i, "mpCount", Script::Converter::fromNumeric(devStats[i].mpCount));
			Script::Struct::setVal(deviceStruct, i, "threadsPerMP", Script::Converter::fromNumeric(devStats[i].threadsPerMP));
			Script::Struct::setVal(deviceStruct, i, "warpSize", Script::Converter::fromNumeric(devStats[i].warpSize));
			Script::Struct::setVal(deviceStruct, i, "maxThreads", Script::Converter::fromNumeric(devStats[i].maxThreads));
		}

		delete[] devStats;
	}
};