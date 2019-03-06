#pragma once

#ifndef IMAGE_PROCESSOR_API
  #error This header must only be included in CWrappers.h
#endif


#include "ScriptCmds/ScriptioMaps.h"
#include "ScriptCmds/LinkageTraitTfms.h"

#define GENERATE_PROC_STUB_PROTOTYPES
  #include "ScriptCmds/GenCommands.h"
#undef GENERATE_PROC_STUB_PROTOTYPES
