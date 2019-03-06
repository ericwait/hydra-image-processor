#pragma once

#ifndef IMAGE_PROCESSOR_API
  #error This header must only be included in CWrappers.h
#endif


#include "../WrapCmds/ScriptioMaps.h"
#include "../WrapCmds/LinkageTraitTfms.h"

#define GENERATE_PROC_STUB_PROTOTYPES
  #include "../WrapCmds/GenCommands.h"
#undef GENERATE_PROC_STUB_PROTOTYPES
