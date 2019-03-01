// This file is used to register commands that are callable via script languages
SCR_BEGIN_COMMANDS
	// These are default commands defined for all script wrappers.
	SCR_CMD_NOPROC(Help, SCR_PARAMS(SCR_OPTIONAL(SCR_SCALAR(std::string), command, "")))
	//SCR_CMD_NOPROC(Info, SCR_PARAMS(SCR_OUTPUT(SCR_STRUCT, cmdInfo)))
	//SCR_CMD_NOPROC(DeviceCount, SCR_PARAMS(SCR_OUTPUT(SCR_SCALAR(uint32_t), numCudaDevices),
	//											SCR_OUTPUT(SCR_STRUCT, memStats)))
	//SCR_CMD_NOPROC(DeviceStats, SCR_PARAMS(SCR_OUTPUT(SCR_STRUCT, deviceStatsArray)))

	// Additional specific wrapped commands should be added here.
	SCR_CMD(Closure, SCR_PARAMS
		(
			SCR_INPUT(SCR_IMAGE(SCR_DYNAMIC), imageIn),
			SCR_OUTPUT(SCR_IMAGE(SCR_DYNAMIC), imageOut),
			SCR_INPUT(SCR_IMAGE_CONVERT(float), kernel),
			SCR_OPTIONAL(SCR_SCALAR(int), numIterations, 1),
			SCR_OPTIONAL(SCR_SCALAR(int), device, -1)
		),
		closure
	)

SCR_END_COMMANDS
