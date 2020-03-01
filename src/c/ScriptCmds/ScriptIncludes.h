#pragma once

#if defined(PY_BUILD)
#include "../Python/PyIncludes.h"
#elif defined(MEX_BUILD)
#include "../Mex/MexIncludes.h"
#else
#error Either PY_BUILD or MEX_BUILD must be defined for project
#endif

#undef max
#undef min

#include "ScriptCmds/HydraConfig.h"

#include "ScriptCmds/ScriptHelpers.h"
#include "ScriptCmds/ScriptCommand.h"
