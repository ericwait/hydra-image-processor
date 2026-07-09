# Extending Hydra is easy
Hydra Image Processor (Hydra) has been written to expedite the addition of functionality. All of the machinery needed to distribute data and iterate over neighborhoods is included. Adding a new function is as easy as copying a template file and replacing the requisite lines of code. This paradigm is intended to encourage research and development of operations that would otherwise be intractable without hardware acceleration. 

# How to contribute
Contribution is as easy as a pull request on GitHub. Following the guidelines will ensure requests are not rejected for minor issues. Once your code conforms to the guidelines, please submit your requests [here](https://github.com/ericwait/hydra-image-processor/pulls). 

Alternatively, you can contribute by detailing your needs on this [forum](https://www.hydraimageprocessor.com/forum/request-functionality). Hydra was also designed to be a practical library that meets the needs of microscopist. I really enjoy when theory and application meet. By explaining in detail (examples are always helpful as well) what your needs are, the community will be able to find novel ways accomplish your goals. 

# Guidelines
## Code is not correct until it is clean

From the very beginning Hydra was written to be clean, consistent, and as clear as possible. The use of object oriented programming and templates have made this project quickly extensible as well as maintainable. Each portion of code, down to the lowest operation, should be as "_glanceable_" as possible. Meaning that use of spacing, letter case, and naming scheme should be as information dense as possible. It all boils down to, make code that assists the reader in their understanding of what it is intended to accomplish.

## Main points to follow

1. Format, format, format. Make it look clean and consistent with existing code.
1. Do not duplicate functionality. Use existing functions/classes when possible. Consider extending existing functionality before creating new.
1. Use templates to ensure code maintainability. 
1. Mimic what as already been done. If code looks inconsistent or "out of place." Correct it or bring it to the community's attention.
1. _**Try crazy things**_! Hydra was built to quickly get operations onto GPU hardware. Use this opportunity to create things that would not otherwise be tractable. 

### These guidelines are intended for GitHub pull requests. Do not let them be a barrier to experimentation. Have FUN!

# Building & testing locally

The Python package builds with [scikit-build-core](https://scikit-build-core.readthedocs.io/):
a single `pip install` configures CMake, compiles the CUDA extension, and installs the
package. You need a CUDA toolkit, a C++ compiler, and a conda environment with the build
tools:

```bash
conda create -n hydra-dev -c conda-forge python numpy cmake ninja scikit-build-core pip
conda activate hydra-dev
pip install . --no-build-isolation -v        # from the repository root

# Smoke test (no GPU required to import)
python -c "import hydra_image_processor as HIP; print(HIP.__version__, HIP.device_count())"
```

The full TIFF accuracy suite (`src/Python/Test`) and the C++ `test_accuracy` binary require a
real NVIDIA GPU and the LFS-tracked ground-truth images (`git lfs pull`). CI runners have no
GPU, so they only verify the package builds and imports; **run the accuracy suite locally on a
GPU box before tagging a release.**

# Releasing

Releases are tag-driven. conda-forge's autotick bot watches for new GitHub releases and opens
a feedstock version-bump PR automatically.

1. Run the full accuracy suite locally on a GPU and confirm it passes.
2. Bump `[project].version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
3. Merge to `main` with a green CI gate.
4. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z` (the tag **must** match the
   `pyproject.toml` version — the release workflow enforces this).
5. The `Release` workflow creates the GitHub Release. Merge the resulting feedstock autotick
   PR (or open one manually), updating the recipe if the build inputs changed.

The conda-forge recipe lives in the
[`hydra-image-processor-feedstock`](https://github.com/conda-forge/hydra-image-processor-feedstock)
repository and is a thin `pip install .` wrapper — keep it free of source patches.
