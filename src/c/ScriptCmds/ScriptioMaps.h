#pragma once

#include "ScriptCommandDefines.h"

#define GENERATE_DEFAULT_IO_MAPPERS
#include "GenCommands.h"
#undef GENERATE_DEFAULT_IO_MAPPERS

/////////////////
// SCRIPT_COMMAND_DEF:
// If necessary, place any non-default input->ouput map definitions here
//   E.g. see EntropyFilter command, for an example of non-standard type-mapping

// EntropyFilter command always returns floats
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, bool)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, uint8_t)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, uint16_t)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, int16_t)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, uint32_t)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, int32_t)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, float)
SCR_DEFINE_IO_TYPE_MAP(EntropyFilter, float, double)


// LoG command always returns floats
SCR_DEFINE_IO_TYPE_MAP(LoG, float, bool)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, uint8_t)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, uint16_t)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, int16_t)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, uint32_t)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, int32_t)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, float)
SCR_DEFINE_IO_TYPE_MAP(LoG, float, double)


// Use largest 'matching' type for sum
SCR_DEFINE_IO_TYPE_MAP(Sum, uint64_t, bool)
SCR_DEFINE_IO_TYPE_MAP(Sum, uint64_t, uint8_t)
SCR_DEFINE_IO_TYPE_MAP(Sum, uint64_t, uint16_t)
SCR_DEFINE_IO_TYPE_MAP(Sum, int64_t, int16_t)
SCR_DEFINE_IO_TYPE_MAP(Sum, uint64_t, uint32_t)
SCR_DEFINE_IO_TYPE_MAP(Sum, int64_t, int32_t)
SCR_DEFINE_IO_TYPE_MAP(Sum, double, float)
SCR_DEFINE_IO_TYPE_MAP(Sum, double, double)


/////////////////
