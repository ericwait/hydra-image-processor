add_custom_command(TARGET HydraMex
    POST_BUILD
    COMMAND ${Matlab_MAIN_PROGRAM} -nosplash -nodisplay -nodesktop $<$<PLATFORM_ID:Windows>:-wait> -r "autoInstallMex('${HYDRA_MODULE_NAME}', '$<TARGET_FILE:HydraMex>');exit;"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/MATLAB/build-scripts
    VERBATIM USES_TERMINAL
)
