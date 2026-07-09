# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hydra Image Processor is a CUDA-accelerated image processing library for 1–5D data (x, y, z, channel, time). Its signature feature is automatic chunking of images larger than GPU memory across one or more GPUs, with halo regions to avoid edge artifacts. A single C++/CUDA backend is exposed identically to Python (`hydra_image_processor` package wrapping a compiled `Hydra` extension) and MATLAB (MEX + generated `.m` wrappers).

An NVIDIA GPU + CUDA Toolkit (12.x in CI) are required to build; tests skip gracefully when no CUDA device is present, but the build itself needs `nvcc`.

## Build Commands

```bash
# Canonical Python build (scikit-build-core drives CMake + nvcc and installs
# the package; run from the repository root in an env with cmake/ninja/numpy)
pip install . --no-build-isolation -v

# Raw CMake targets (Linux/macOS-style; CUDA toolkit must be on PATH)
cmake -S . -B build
cmake --build build --config Release --target HydraPy        # Python extension
cmake --build build --config Release --target HydraMex       # MATLAB MEX (needs MATLAB)
cmake --build build --config Release --target test_accuracy  # C++ accuracy tests

# Windows/MSVC preset (the only preset defined)
cmake --preset VS64
cmake --build --preset VS64-debug
```

- `-DHYDRA_MODULE_NAME=HIP` renames the output module (default `Hydra`; CI uses `HIP` for MATLAB builds).
- The `HydraPy` target writes `Hydra.pyd`/`Hydra.so` directly into `src/Python/hydra_image_processor/`.
- Building `HydraMex` triggers a post-build MATLAB step (`src/c/Mex/autoBuildMex.cmake`) that regenerates the `.m` wrapper files — never hand-edit generated wrappers in `src/MATLAB/+Hydra/@Cuda/`.
- `-DCMAKE_CUDA_ARCHITECTURES` defaults to `all-major`; use `86`/`native` for fast local iteration.
- End-user distribution is via **conda-forge** (`conda install -c conda-forge hydra-image-processor`); the recipe lives in the separate `hydra-image-processor-feedstock` repository and must stay a patch-free `pip install .` wrapper.
- `-DUSE_PROCESS_MUTEX=ON` enables a cross-process GPU mutex (boost interprocess, vendored in `src/c/external/`).

## Tests

```bash
# C++ accuracy tests (needs a CUDA device at runtime; skips otherwise)
cmake --build build --target test_accuracy
ctest --test-dir build                 # runs CppAccuracyTest
./build/src/c/test_back/test_accuracy  # or run the binary directly

# Python smoke tests (plain scripts, no pytest)
cd src/Python && python test_package.py

# Python TIFF accuracy suite (needs a GPU + tifffile; THE release gate, see TESTING.md)
python src/Python/Test/test_accuracy.py

# MATLAB accuracy tests (matlab.unittest; compares against test_data/*.tif)
matlab -batch "results = runtests('src/MATLAB/+Test/AccuracyTest.m')"
```

- C++ tests use a minimal custom framework (`TEST_ASSERT`/`RUN_TEST` in `src/c/test_back/test_accuracy.cpp`) and call the generated `CudaCall_<Name>::run(...)` entry points directly. To run a single test, comment out other `RUN_TEST` lines or add a new `RUN_TEST` — there is no filter flag.
- Ground-truth images live in `test_data/` named `<Op>_c<channel>_<params>.tif`. They are **Git LFS** files (`.gitattributes` tracks `*.tif *.png *.mat *.pyd *.so *.mex*`) — run `git lfs pull` before running accuracy tests, and any new binary test assets go through LFS.

## Architecture

### Macro-driven command registration (the heart of the codebase)

Every operation is declared **once** as an `SCR_CMD(Name, SCR_PARAMS(...), cudaFunc)` line in `src/c/ScriptCmds/ScriptCommands.h`. That table is re-included repeatedly by `src/c/ScriptCmds/GenCommands.h` under different `GENERATE_*` preprocessor passes, which synthesize:

- an argument parser and command class per operation (`GENERATE_SCRIPT_COMMANDS`),
- concrete `CudaCall_<Name>::run(...)` stubs for each of the 8 supported pixel types (`GENERATE_PROC_STUBS`, driven from `src/c/Cuda/CWrapperAutogen.cu`),
- input→output pixel-type maps (`GENERATE_DEFAULT_IO_MAPPERS`, overridable via `SCR_DEFINE_IO_TYPE_MAP`),
- help strings and the runtime command map used by both frontends (`GENERATE_CONSTEXPR_MEM`, `GENERATE_COMMAND_MAP`).

`src/c/ScriptCmds/ScriptCommandModule.h` is the instantiation point and must be included in **exactly one** `.cpp` per module (`src/c/Python/PyCommandModule.cpp`, `src/c/Mex/MexCommandModule.cpp`). Shared per-command runtime logic (arg conversion → pixel-type dispatch → output allocation → CUDA call) lives in `src/c/ScriptCmds/ScriptCommandImpl.h`.

The 8 supported pixel types (`bool, uint8, uint16, int16, uint32, int32, float, double`) are hard-coded in three places that must stay in sync: the runtime dispatch in `ScriptCommandImpl.h`, the stub generators in `GenCommands.h`, and the type transforms in `src/c/ScriptCmds/LinkageTraitTfms.h`.

### Frontends

Python and MATLAB compile the same `HydraCudaStatic` backend and command framework; they differ only by a `PY_BUILD`/`MEX_BUILD` define and a language-specific ArgConverter (`src/c/ScriptCmds/PyArgConverter.h` / `MexArgConverter.h`).

- **Python**: `src/c/Python/HydraPyModule.cpp` builds the `PyMethodDef` table by iterating the generated command map. The pure-Python layer (`src/Python/hydra_image_processor/`) follows a GPU-first/CPU-fallback pattern: `core.py` wraps `cuda/core.py` (calls the compiled extension) with fallbacks in `local/core.py`.
- **MATLAB**: `src/c/Mex/HydraMexModule.cpp` is a single `mexFunction` dispatching on a command-name string. User-facing `.m` files under `src/MATLAB/+Hydra/` are generated at build time by `src/MATLAB/build-scripts/BuildMexClass.m` + `autoInstallMex.m` from the C++ help strings (`+HIP` is the legacy package name).

### CUDA core and chunking

`src/c/Cuda/ImageChunk.cpp` computes chunks that fit the smallest GPU's free memory, including kernel-halo margins, and picks split axes to minimize overlap recomputation. Host drivers (e.g. `src/c/Cuda/CudaGaussian.cuh`) follow a standard pattern: build kernels → `calculateBuffers(...)` → OpenMP parallel with one thread per GPU → per chunk `sendROI` → launch `__global__` kernel(s) via ping-pong buffers (`CudaDeviceImages.cuh`) → `retriveROI`. Device-side images (`CudaImageContainer.cuh`) are passed by value into kernels; `KernelIterator.cuh` walks structuring-element neighborhoods. Host-side images are non-owning `ImageView<T>` (`src/c/Cuda/ImageView.h`).

### Adding a new operation

1. Write `src/c/Cuda/CudaFoo.cuh` with the `__global__` kernel and a host driver `cFoo<PixelTypeIn, PixelTypeOut>(ImageView ...)`; include it in `src/c/Cuda/CWrapperAutogen.cu`.
2. Add `src/c/ScriptCmds/Commands/ScrCmdFoo.h` (`SCR_COMMAND_CLASSDEF(Foo)` + `SCR_HELP_STRING`); include it in `ScriptCommandModule.h`.
3. Add one `SCR_CMD(Foo, SCR_PARAMS(...), cFoo)` line to `src/c/ScriptCmds/ScriptCommands.h`.

Type dispatch, Python method registration, and MATLAB wrappers are all generated from that. Only the pretty-named Python wrappers in `src/Python/hydra_image_processor/` (`cuda/core.py`, `core.py`) are added by hand.

## Branching & releases

- `main` is protected (PR + green CI required); `develop` is the integration branch.
- Work on `feature/<name>` branches cut from `develop`, merge them into `develop`, and open a PR `develop` → `main` when a release-worthy state is reached.
- Releases are tag-driven: bump `pyproject.toml` + `CHANGELOG.md`, work through TESTING.md (the GPU accuracy suites cannot run in CI), merge to `main`, then `git tag vX.Y.Z && git push origin vX.Y.Z`. `release.yml` enforces tag == version and publishes the GitHub Release; conda-forge's autotick bot then opens a feedstock version-bump PR.

## CI

`.github/workflows/ci.yml` is the build-gate: on Windows + Linux it installs CUDA/compilers from conda-forge, builds via the same `pip install .` path the feedstock uses, and runs an import smoke test (runners have no GPU — accuracy suites run locally, see TESTING.md). `.github/workflows/release.yml` runs on `v*` tags. `.github/workflows/matlab-multibuild.yml` builds `HIP` MEX binaries on Windows + Linux and packages the MATLAB toolbox from `src/MATLAB/HydraImageProcessor.prj`.
