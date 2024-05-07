# Setup MEX interface if Matlab was found
add_library(HydraPy MODULE "")

# Require c++11 and set build definition to PY_BUILD
target_compile_features(HydraPy PRIVATE cxx_std_11)
target_compile_definitions(HydraPy PRIVATE PY_BUILD)

# Set USE_PROCESS_MUTEX if cmake variable is set
if ( USE_PROCESS_MUTEX )
    target_compile_definitions(HydraPy PRIVATE USE_PROCESS_MUTEX)
endif()

# Link against Python and NumPy libraries
target_link_libraries(HydraPy PRIVATE HydraCudaStatic Python::Python Python::NumPy)

# Change output library name to Hydra.<dll/so/dylib>
set_target_properties(HydraPy
    PROPERTIES
        OUTPUT_NAME ${HYDRA_MODULE_NAME}
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/Python>
)

# On windows specifically set the suffix to .pyd
if ( WIN32 )
    set_target_properties(HydraPy PROPERTIES SUFFIX .pyd)
endif()

# Setup Python/NumPy include directories
target_include_directories(HydraPy PRIVATE Python::Python Python::NumPy)

# Setup src include directories
target_include_directories(HydraPy
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../ScriptCmds>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../external>
)

target_sources(HydraPy
    PRIVATE
        PyArgConverter.h
        PyIncludes.h
        PyTypes.h

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
        HydraPyModule.cpp
        PyCommandModule.cpp
)
