setlocal EnableDelayedExpansion

:: Create a unique build directory for conda
mkdir conda-build-dir
cd conda-build-dir

:: Configure CMake
:: We need to point to the root CMakeLists.txt
cmake -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPython3_EXECUTABLE="%PYTHON%" ^
    -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120" ^
    "%SRC_DIR%"
if errorlevel 1 exit 1

:: Build the HydraPy target
cmake --build . --config Release --target HydraPy
if errorlevel 1 exit 1

:: The build should have placed Hydra.pyd into src/Python/hydra_image_processor
:: Verify it exists (optional but good for debugging)
if not exist "%SRC_DIR%\src\Python\hydra_image_processor\Hydra.pyd" (
    echo "Error: Hydra.pyd not found in expected location!"
    exit 1
)

:: Install the Python package
cd "%SRC_DIR%\src\Python"
"%PYTHON%" -m pip install . --no-deps --ignore-installed -v
if errorlevel 1 exit 1
