// This file is used to register commands that are callable via script languages
BEGIN_WRAP_COMMANDS
	// These are default commands defined for all wrapper dlls.
	DEF_WRAP_COMMAND(Info)
	DEF_WRAP_COMMAND(Help)
	DEF_WRAP_COMMAND(DeviceCount)
	DEF_WRAP_COMMAND(DeviceStats)
	// Additional specific wrapped commands should be added here.
	DEF_WRAP_COMMAND(Closure)
	DEF_WRAP_COMMAND(ElementWiseDifference)
	DEF_WRAP_COMMAND(EntropyFilter)
	DEF_WRAP_COMMAND(Gaussian)
	DEF_WRAP_COMMAND(GetMinMax)
	DEF_WRAP_COMMAND(HighPassFilter)
	DEF_WRAP_COMMAND(LoG)
	DEF_WRAP_COMMAND(MaxFilter)
	DEF_WRAP_COMMAND(MeanFilter)
	DEF_WRAP_COMMAND(MedianFilter)
	DEF_WRAP_COMMAND(MinFilter)
	DEF_WRAP_COMMAND(MinMax)
	DEF_WRAP_COMMAND(MultiplySum)
	DEF_WRAP_COMMAND(Opener)
	DEF_WRAP_COMMAND(StdFilter)
	DEF_WRAP_COMMAND(Sum)
	DEF_WRAP_COMMAND(VarFilter)
	DEF_WRAP_COMMAND(WienerFilter)
END_WRAP_COMMANDS
