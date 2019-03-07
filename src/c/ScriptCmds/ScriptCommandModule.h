#pragma once
/////////////////
// This is the implementation file for all script commands and
// should be included in e.g. PyCommand.cpp
//
// NOTE: Do NOT include this in more than one cpp file per-project

#include "../Cuda/CWrappers.h"

#include "ScriptIncludes.h"
#include "ScriptCommandImpl.h"

#if defined(PY_BUILD)
#include "PyArgConverter.h"
#elif defined (MEX_BUILD)
#include "MexArgConverter.h"
#else
#error Either PY_BUILD or MEX_BUILD must be defined for project
#endif


#define GENERATE_SCRIPT_COMMANDS
#include "ScriptCmds/GenCommands.h"
#undef GENERATE_SCRIPT_COMMANDS


/////////////////
// SCRIPT_COMMAND_DEF:
// Place includes for all script command class definitions here

// Special helper commands for Mex interface
#include "Commands/ScrCmdHelp.h"
#include "Commands/ScrCmdInfo.h"

// Special Cuda device information commands
#include "Commands/ScrCmdDeviceCount.h"
#include "Commands/ScrCmdDeviceStats.h"

// Cuda processing commands
#include "Commands/ScrCmdClosure.h"
#include "Commands/ScrCmdElementWiseDifference.h"
#include "Commands/ScrCmdEntropyFilter.h"
#include "Commands/ScrCmdGaussian.h"
#include "Commands/ScrCmdGetMinMax.h"
#include "Commands/ScrCmdHelp.h"
#include "Commands/ScrCmdHighPassFilter.h"
#include "Commands/ScrCmdIdentityFilter.h"
#include "Commands/ScrCmdInfo.h"
#include "Commands/ScrCmdLog.h"
#include "Commands/ScrCmdMaxFilter.h"
#include "Commands/ScrCmdMeanFilter.h"
#include "Commands/ScrCmdMedianFilter.h"
#include "Commands/ScrCmdMinFilter.h"
#include "Commands/ScrCmdMultiplySum.h"
#include "Commands/ScrCmdOpener.h"
#include "Commands/ScrCmdStdFilter.h"
#include "Commands/ScrCmdSum.h"
#include "Commands/ScrCmdVarFilter.h"
#include "Commands/ScrCmdWienerFilter.h"

/////////////////

/////////////////
// Creates memory for constexpr variables and
// initializes ScriptCommand::m_commands list
//
// NOTE: This will cause compile errors if included
//   in more than one cpp module per-project
#define GENERATE_CONSTEXPR_MEM
#include "GenCommands.h"
#undef GENERATE_CONSTEXPR_MEM

#define GENERATE_COMMAND_MAP
#include "GenCommands.h"
#undef GENERATE_COMMAND_MAP
