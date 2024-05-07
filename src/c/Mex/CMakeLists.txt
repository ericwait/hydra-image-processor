# Setup MEX interface if Matlab was found
add_library(HydraMex MODULE "")

# Set up custom post-build command to run MATLAB build scripts
include(autoBuildMex.cmake)

# Require c++11 and set build definition to MEX_BUILD
target_compile_features(HydraMex PRIVATE cxx_std_11)
target_compile_definitions(HydraMex PRIVATE MEX_BUILD)

# Set USE_PROCESS_MUTEX if cmake variable is set
if ( USE_PROCESS_MUTEX )
    target_compile_definitions(HydraMex PRIVATE USE_PROCESS_MUTEX)
endif()

# Link against Matlab libraries
target_link_libraries(HydraMex PRIVATE HydraCudaStatic ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY})

# Change output library name to <ModuleName>.mex<plaform>
set_target_properties(HydraMex
    PROPERTIES
        OUTPUT_NAME ${HYDRA_MODULE_NAME}
        PREFIX ""
        SUFFIX .${Matlab_MEX_EXTENSION}
        POSITION_INDEPENDENT_CODE ON
)

# Setup Matlab include directories
target_include_directories(HydraMex PRIVATE ${Matlab_INCLUDE_DIRS})

# Setup src include directories
target_include_directories(HydraMex
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../ScriptCmds>
)

# Include module definition file for Windows
if ( WIN32 )
    target_sources(HydraMex PRIVATE HydraMex.def)
endif()

target_sources(HydraMex
    PRIVATE
        MexArgConverter.h
        MexIncludes.h
        MexTypes.h

        ../mph/const_string.h
        ../mph/integer_sequence.h
        ../mph/preproc_helper.h
        ../mph/qualifier_helpers.h
        ../mph/tuple_helpers.h

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

        ../ScriptCmds/ScopedProcessMutex.cpp
        MexCommandModule.cpp
        HydraMexModule.cpp
)
