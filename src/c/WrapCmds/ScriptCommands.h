// This file is used to register commands that are callable via script languages
BEGIN_SCRIPT_COMMANDS
	// These are default commands defined for all script wrappers.
	DEF_SCRIPT_COMMAND(Info, DEF_PARAMS(SCRIPT_OUT()))
	DEF_SCRIPT_COMMAND(Help, )
	DEF_SCRIPT_COMMAND(DeviceCount)
	DEF_SCRIPT_COMMAND(DeviceStats)
	// Additional specific wrapped commands should be added here.
	DEF_SCRIPT_COMMAND(Closure)
	DEF_SCRIPT_COMMAND(ElementWiseDifference)
	DEF_SCRIPT_COMMAND(EntropyFilter)
	DEF_SCRIPT_COMMAND(Gaussian)
	DEF_SCRIPT_COMMAND(GetMinMax)
	DEF_SCRIPT_COMMAND(HighPassFilter)
	DEF_SCRIPT_COMMAND(IdentityFilter)
	DEF_SCRIPT_COMMAND(LoG)
	DEF_SCRIPT_COMMAND(MaxFilter)
	DEF_SCRIPT_COMMAND(MeanFilter)
	DEF_SCRIPT_COMMAND(MedianFilter)
	DEF_SCRIPT_COMMAND(MinFilter)
	DEF_SCRIPT_COMMAND(MultiplySum)
	DEF_SCRIPT_COMMAND(Opener)
	DEF_SCRIPT_COMMAND(StdFilter)
	DEF_SCRIPT_COMMAND(Sum)
	DEF_SCRIPT_COMMAND(VarFilter)
	DEF_SCRIPT_COMMAND(WienerFilter)
END_SCRIPT_COMMANDS
