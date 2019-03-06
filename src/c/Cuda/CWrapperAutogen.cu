#include "CWrappers.h"

// TODO: Unify cuda filter includes into single header?

#include "CudaClosure.cuh"

// Autogenerate all stub calls to cuda backends
#define GENERATE_PROC_STUBS
#include "../WrapCmds/GenCommands.h"
#undef GENERATE_PROC_STUBS
