add_custom_command(TARGET HydraMex
    POST_BUILD
    COMMAND ${Matlab_MAIN_PROGRAM} -nosplash $<$<NOT:$<PLATFORM_ID:Windows>>:-nodisplay> -nodesktop $<$<VERSION_LESS:${Matlab_VERSION_STRING},9.6>:$<$<PLATFORM_ID:Windows>:-wait>> $<IF:$<VERSION_LESS:${Matlab_VERSION_STRING},9.6>,-r,-batch> "autoInstallMex('${HYDRA_MODULE_NAME}', '$<TARGET_FILE:HydraMex>')$<$<VERSION_LESS:${Matlab_VERSION_STRING},9.6>:$<SEMICOLON>exit>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/MATLAB/build-scripts
    VERBATIM USES_TERMINAL
)