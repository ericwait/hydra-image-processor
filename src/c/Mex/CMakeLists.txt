# ===========================
# Hydra MEX Interface
# ===========================
add_library(HydraMex MODULE)

# Require modern C++ and define for MEX
target_compile_features(HydraMex PRIVATE cxx_std_17)
target_compile_definitions(HydraMex PRIVATE MEX_BUILD)

if (USE_PROCESS_MUTEX)
    target_compile_definitions(HydraMex PRIVATE USE_PROCESS_MUTEX)
endif()

# Output name should be <ModuleName>.mex<platform>
if (NOT Matlab_MEX_EXTENSION)
    message(FATAL_ERROR "Matlab_MEX_EXTENSION is not set")
endif()

message(STATUS "Matlab MEX extension: '${Matlab_MEX_EXTENSION}'")

set_target_properties(HydraMex PROPERTIES
    OUTPUT_NAME "${HYDRA_MODULE_NAME}"
    PREFIX ""
    SUFFIX ".${Matlab_MEX_EXTENSION}"
    POSITION_INDEPENDENT_CODE ON
)

# ===========================
# Includes and Linking
# ===========================
target_include_directories(HydraMex
    PRIVATE
        ${Matlab_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../ScriptCmds>
)

target_link_libraries(HydraMex
    PRIVATE HydraCudaStatic
    PRIVATE ${Matlab_MX_LIBRARY}
    PRIVATE ${Matlab_MEX_LIBRARY}
)

# ===========================
# MEX Sources
# ===========================
target_sources(HydraMex
    PRIVATE
        # Local headers
        MexArgConverter.h
        MexIncludes.h
        MexTypes.h

        # Meta programming helpers
        ../mph/const_string.h
        ../mph/integer_sequence.h
        ../mph/preproc_helper.h
        ../mph/qualifier_helpers.h
        ../mph/tuple_helpers.h

        # Script command infrastructure
        ../ScriptCmds/ArgConverter.h
        ../ScriptCmds/GenCommands.h
        ../ScriptCmds/HydraConfig.h
        ../ScriptCmds/LinkageTraitTfms.h
        ../ScriptCmds/ScopedProcessMutex.h
        ../ScriptCmds/ScriptCommand.h
        ../ScriptCmds/ScriptCommandDefines.h
        ../ScriptCmds/ScriptCommandImpl.h
        ../ScriptCmds/ScriptCommandModule.h
        ../ScriptCmds/ScriptCommands.h
        ../ScriptCmds/ScriptHelpers.h
        ../ScriptCmds/ScriptIncludes.h
        ../ScriptCmds/ScriptioMaps.h
        ../ScriptCmds/ScriptTraits.h
        ../ScriptCmds/ScriptTraitTfms.h

        # Script command implementations
        ../ScriptCmds/Commands/ScrCmdCheckConfig.h
        ../ScriptCmds/Commands/ScrCmdClosure.h
        ../ScriptCmds/Commands/ScrCmdDeviceCount.h
        ../ScriptCmds/Commands/ScrCmdDeviceStats.h
        ../ScriptCmds/Commands/ScrCmdElementWiseDifference.h
        ../ScriptCmds/Commands/ScrCmdEntropyFilter.h
        ../ScriptCmds/Commands/ScrCmdGaussian.h
        ../ScriptCmds/Commands/ScrCmdGetMinMax.h
        ../ScriptCmds/Commands/ScrCmdHelp.h
        ../ScriptCmds/Commands/ScrCmdHighPassFilter.h
        ../ScriptCmds/Commands/ScrCmdIdentityFilter.h
        ../ScriptCmds/Commands/ScrCmdInfo.h
        ../ScriptCmds/Commands/ScrCmdLoG.h
        ../ScriptCmds/Commands/ScrCmdMaxFilter.h
        ../ScriptCmds/Commands/ScrCmdMeanFilter.h
        ../ScriptCmds/Commands/ScrCmdMedianFilter.h
        ../ScriptCmds/Commands/ScrCmdMinFilter.h
        ../ScriptCmds/Commands/ScrCmdMultiplySum.h
        ../ScriptCmds/Commands/ScrCmdNLMeans.h
        ../ScriptCmds/Commands/ScrCmdOpener.h
        ../ScriptCmds/Commands/ScrCmdStdFilter.h
        ../ScriptCmds/Commands/ScrCmdSum.h
        ../ScriptCmds/Commands/ScrCmdVarFilter.h
        ../ScriptCmds/Commands/ScrCmdWienerFilter.h

        # Source files
        ../ScriptCmds/ScopedProcessMutex.cpp
        MexCommandModule.cpp
        HydraMexModule.cpp
)

# Add Windows-specific .def file if needed
if (WIN32)
    target_sources(HydraMex PRIVATE HydraMex.def)
endif()

# ===========================
# Post-Build MATLAB Autogen
# ===========================
include(${CMAKE_CURRENT_LIST_DIR}/autoBuildMex.cmake)
