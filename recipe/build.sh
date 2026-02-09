#!/bin/bash

mkdir -p conda-build-dir
cd conda-build-dir

# Configure CMake
cmake ${CMAKE_ARGS} -G "Ninja" 
    -DCMAKE_BUILD_TYPE=Release 
    -DPython3_EXECUTABLE="$PYTHON" 
    -DCMAKE_INSTALL_PREFIX="$PREFIX" 
    "$SRC_DIR"

# Build the HydraPy target
cmake --build . --config Release --target HydraPy

# Install the Python package
cd "$SRC_DIR/src/Python"
"$PYTHON" -m pip install . --no-deps --ignore-installed -v
