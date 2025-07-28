# ===========================
# Hydra Python Interface
# ===========================

add_library(HydraPy MODULE)

# Require modern C++ and define for Python build
target_compile_features(HydraPy PRIVATE cxx_std_17)
target_compile_definitions(HydraPy PRIVATE PY_BUILD)

if (USE_PROCESS_MUTEX)
    target_compile_definitions(HydraPy PRIVATE USE_PROCESS_MUTEX)
endif()

# Output name should be Hydra.<suffix> (e.g., .pyd on Windows)
set_target_properties(HydraPy PROPERTIES
    OUTPUT_NAME ${HYDRA_MODULE_NAME}
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Python
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Python
)

# On Windows, set the suffix to .pyd for Python modules
if (WIN32)
    set_target_properties(HydraPy PROPERTIES SUFFIX ".pyd")
endif()

# ===========================
# Includes and Linking
# ===========================
target_include_directories(HydraPy
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../ScriptCmds>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../external>
)

target_link_libraries(HydraPy
    PRIVATE HydraCudaStatic
    PRIVATE Python::Python
    PRIVATE Python::NumPy
)

# ===========================
# Python Binding Sources
# ===========================
target_sources(HydraPy
    PRIVATE
        # Local headers
        PyArgConverter.h
        PyIncludes.h
        PyTypes.h

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
        HydraPyModule.cpp
        PyCommandModule.cpp
)
