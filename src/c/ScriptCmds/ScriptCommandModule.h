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

#include "ScrCmdHelp.h"
#include "ScrCmdInfo.h"

#include "ScrCmdDeviceCount.h"
#include "ScrCmdDeviceStats.h"

#include "ScrCmdClosure.h"

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
