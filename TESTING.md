# Release testing checklist

Pre-release verification for `hydra-image-processor`. CI is **build-only** (runners have no
GPU), so the GPU correctness steps must be run locally before tagging. Work top to bottom;
each phase builds on the previous one. See [CONTRIBUTING.md](CONTRIBUTING.md) for the release
flow this feeds into.

## A. Local build from a clean tree
- [ ] Clean tree: `git status` shows no stray build artifacts (e.g. a leftover `Hydra.pyd`).
- [ ] Fresh build from the repo root, inside a VS dev shell, in a clean conda env:
      `pip install . --no-build-isolation -v`
- [ ] Build log shows the wheel layout: `hydra_image_processor/Hydra.pyd`, top-level `HIP.py`,
      top-level `Hydra.py`, and `hydra_image_processor-4.0.0-...whl`.
- [ ] Building for all archs (closer to what conda-forge ships) also works:
      `set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=all-major` then reinstall.

## B. Import & API surface
- [ ] `python -c "import hydra_image_processor as HIP; print(HIP.__version__)"` → `4.0.0`
- [ ] `import HIP` and `import Hydra` both work and are the same object as
      `hydra_image_processor.Hydra`.
- [ ] Legacy raw API reachable via shim (`HIP.Gaussian(...)`, capitalized) **and** the
      pythonic API (`hydra_image_processor.gaussian(...)`, lowercase).

## C. GPU correctness — the part CI cannot do
- [ ] `git lfs pull` to fetch the real ground-truth images (the GitHub tarball has only
      LFS pointers, not the binaries).
- [ ] Build and run the C++ accuracy test (`-DHYDRA_BUILD_TESTS=ON`, target `test_accuracy`)
      → `Test Summary: PASSED`.
- [ ] Run the Python TIFF accuracy suite (needs `tifffile`):
      `python src/Python/Test/test_accuracy.py` → `Test Summary: PASSED`.
      **This is the gate for tagging.**
- [ ] Spot-check one op end-to-end (e.g. `gaussian` on a small array) and `device_count() >= 1`.

## D. Working-tree hygiene
- [ ] After building, `git status` is still clean — the in-source `Hydra.pyd` is ignored:
      `git check-ignore src/Python/hydra_image_processor/Hydra.pyd` prints the path.
- [ ] `git log --oneline` shows the build/ci/docs commits; the `Hydra.mexw64` change remains
      separate (commit or discard it intentionally).

## E. CI workflows (GitHub)
- [ ] Push the working branch → `ci.yml` runs and **both** `ubuntu-latest` and `windows-latest`
      jobs build and pass the import smoke test (`__version__ == 4.0.0`, `HIP is Hydra`).
- [ ] Open a PR to `main` → the gate runs on the PR.
- [ ] `release.yml` version guard sanity: the tag (minus `v`) must equal the `pyproject.toml`
      version. (An `rc` tag won't match unless `pyproject` is set to that rc.)

## F. Feedstock dry-run (proves the patches are truly gone)
- [ ] In `hydra-image-processor-feedstock` on branch `prepare-v4.0.0`, temporarily point
      `source` at the local checkout / branch tarball, then run `python build-locally.py`
      (or `pixi run`) for one Windows variant → builds **with no patches** and the
      `import hydra_image_processor` test passes.

## G. Release rehearsal (when A–F are green)
- [ ] Tag `v4.0.0` and push → `release.yml` passes the version guard and creates the GitHub
      Release.
- [ ] Update `source.sha256` in the feedstock `prepare-v4.0.0` recipe to the `v4.0.0` tarball
      hash; push and open the feedstock PR (or let the autotick bot open one and graft the
      recipe changes onto it).
- [ ] Feedstock CI (Azure/GitHub) builds all variants green.

## H. End-user install (post-publish)
- [ ] In a fresh env, each installer resolves and imports:
      `conda install -c conda-forge hydra-image-processor`, `mamba install ...`,
      `pixi add hydra-image-processor`
      → `python -c "import hydra_image_processor as HIP; print(HIP.__version__)"`.
