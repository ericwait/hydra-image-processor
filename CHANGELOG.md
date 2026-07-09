# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [4.0.0] - 2026-07-08

### Changed
- **Repackaged the Python distribution.** The library is imported as the
  `hydra_image_processor` package (conventionally `import hydra_image_processor as HIP`),
  which wraps the compiled `Hydra` CUDA extension and exposes a Pythonic, snake_case API
  (e.g. `gaussian`, `device_count`).
- **Unified the build on [scikit-build-core].** A single `pip install .` now drives CMake,
  builds the extension, and installs the package — the same path used locally, in CI, and by
  the conda-forge feedstock. This retires the recipe-side patches the feedstock had to carry.
- Made the source self-sufficient for conda-forge: CUDA C++ standard raised to 17, hardcoded
  GPU architectures removed in favor of `CMAKE_CUDA_ARCHITECTURES` (default `all-major`), and
  a package-relative install target added.
- Single source of truth for the version (`pyproject.toml`); `__version__` is now read from
  the installed package metadata.

### Added
- Top-level `HIP` and `Hydra` compatibility shims so legacy `import HIP` / `import Hydra`
  code keeps working.
- Windows + Linux CI build-gate (`.github/workflows/ci.yml`) and a tag-driven release
  workflow with a version guard (`.github/workflows/release.yml`).
- Python TIFF accuracy suite (`src/Python/Test/test_accuracy.py`) mirroring the MATLAB
  `AccuracyTest.m`; it is the GPU-correctness gate for releases (see TESTING.md).

### Fixed
- `make_ellipsoid_mask` produced an off-center, asymmetric structuring element (shifted one
  voxel), so morphological ops using it diverged from the MATLAB/C++ results.
- The command-framework headers now include `<cstdint>` explicitly instead of relying on
  transitive includes, fixing the build with newer gcc/libstdc++ toolchains.

### Removed
- The in-repo conda-build recipe (`recipe/`) and the anaconda.org upload workflow.
  Distribution is now solely via conda-forge.

[scikit-build-core]: https://scikit-build-core.readthedocs.io/
