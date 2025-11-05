@echo off
REM Script to build Doxygen documentation for Hydra Image Processor

echo ========================================
echo Hydra Image Processor - Documentation Build
echo ========================================
echo.

REM Check if Doxygen is installed
where doxygen >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Doxygen is not installed
    echo.
    echo Please install Doxygen from:
    echo   https://www.doxygen.nl/download.html
    echo.
    pause
    exit /b 1
)

REM Display Doxygen version
for /f "delims=" %%i in ('doxygen --version') do set DOXYGEN_VERSION=%%i
echo Using Doxygen version: %DOXYGEN_VERSION%
echo.

REM Check if Graphviz/dot is installed (optional but recommended)
where dot >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%i in ('dot -V 2^>^&1') do set DOT_VERSION=%%i
    echo Using Graphviz: !DOT_VERSION!
) else (
    echo Warning: Graphviz (dot) not found. Graphs will not be generated.
    echo   Download from: https://graphviz.org/download/
)
echo.

REM Check if Doxyfile exists
if not exist "Doxyfile" (
    echo Error: Doxyfile not found in current directory
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Clean previous documentation
if exist "docs\html" (
    echo Cleaning previous documentation...
    rmdir /s /q docs\html 2>nul
    rmdir /s /q docs\latex 2>nul
    rmdir /s /q docs\xml 2>nul
    rmdir /s /q docs\man 2>nul
)

REM Build documentation
echo Building documentation...
doxygen Doxyfile

REM Check if build was successful
if exist "docs\html\index.html" (
    echo.
    echo ========================================
    echo Documentation built successfully!
    echo ========================================
    echo.
    echo View the documentation by opening:
    echo   docs\html\index.html
    echo.
    echo Opening in default browser...
    start docs\html\index.html
) else (
    echo Error: Documentation build failed
    pause
    exit /b 1
)

pause
