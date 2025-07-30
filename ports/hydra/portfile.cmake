vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO ericwait/hydra-image-processor
    REF v${GITVERSION_SEMVER} # you can pass this in via `cmake -D`
    SHA512 0 # This must be updated
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    PREFER_NINJA
)

vcpkg_cmake_build()

vcpkg_cmake_install()

vcpkg_cmake_config_fixup()

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)