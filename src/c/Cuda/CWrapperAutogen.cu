#include "CWrappers.h"

// TODO: Unify cuda filter includes into single header?
#include "CudaClosure.cuh"
#include "CudaElementWiseDifference.cuh"
#include "CudaEntropyFilter.cuh"
#include "CudaGaussian.cuh"
#include "CudaGetMinMax.cuh"
#include "CudaHighPassFilter.cuh"
#include "CudaIdentityFilter.cuh"
#include "CudaLoG.cuh"
#include "CudaMaxFilter.cuh"
#include "CudaMedianFilter.cuh"
#include "CudaMeanFilter.cuh"
#include "CudaMinFilter.cuh"
#include "CudaMinMax.cuh"
#include "CudaMultiplySum.cuh"
#include "CudaOpener.cuh"
#include "CudaStdFilter.cuh"
#include "CudaSum.cuh"
#include "CudaVarFilter.cuh"
#include "CudaWienerFilter.cuh"
#include "CudaNLMeans.cuh"

// Autogenerate all stub calls to cuda backends
#define GENERATE_PROC_STUBS
#include "ScriptCmds/GenCommands.h"
#undef GENERATE_PROC_STUBS
