# Hydra Image Processor Roadmap

This document assesses how far the current implementation is from the project's ideals and lays out the agreed plan of record:
five ordered steps, plus later phases, executed on feature branches that PR into `develop`.
It was last grounded against the codebase in July 2026 (commit `c7f1f07`); file paths and claims reference that state.
The re-arrangeable operation backlog referenced by Step 5 lives in [BACKLOG.md](BACKLOG.md).

## Vision

Hydra aims to be:

1. A library of **fast, GPU-accelerated filters** usable from multiple languages (C/C++, MATLAB, Python, C#, ...).
2. **Energy-isolated at image boundaries**: kernels are renormalized at edges instead of padding/reflecting,
   so no energy is invented or lost at borders.
3. **Runtime-hardware-aware**: images larger than GPU memory are chunked automatically,
   and work is distributed across multiple GPUs when available.

Parts of this vision are already real and load-bearing:

- The energy-insulated boundary exists: `KernelIterator` (`src/c/Cuda/KernelIterator.cu`) clips the kernel window at image borders,
  and `cudaMultiplySum` (`src/c/Cuda/CudaMultiplySum.cuh`) accumulates the in-bounds kernel weight and renormalizes.
- Chunking and multi-GPU distribution exist: `src/c/Cuda/ImageChunk.cpp` sizes chunks (including kernel-halo margins)
  to the smallest GPU's free memory, and every host driver stripes chunks across one OpenMP thread per GPU.
- The multi-language architecture exists: every operation is declared once in `src/c/ScriptCmds/ScriptCommands.h`,
  and X-macro passes in `GenCommands.h` generate the parsers, pixel-type dispatch, help text,
  and command tables consumed by both the Python and MATLAB frontends.

The gaps are in **distribution** (nothing is installable from a public registry today),
**reach** (no CPU path, no C ABI, NVIDIA-only),
and **throughput features** (no FFT, no cross-call GPU residency, no GPU-direct I/O).

## Gap assessment

| Goal | Current state | Distance |
| --- | --- | --- |
| conda / mamba / pixi / uv | Working-ish recipe with Windows-only CI uploading to an anaconda.org channel. Not on conda-forge. Nothing on PyPI, so uv has nothing to install. Known bugs listed in Step 0. | Medium |
| vcpkg / conan | Unmerged `feature/vcpkg` branch has near-complete CMake install/export and a port skeleton with a `SHA512 0` placeholder. Mainline has zero `install()` rules, no `project(VERSION)`. | Medium — salvage and finish |
| CPU fallback | C++: none — the build fails without the CUDA toolkit. Python: 100% `NotImplementedError` stubs. MATLAB: ~14/19 real toolbox fallbacks, but with divergent boundary behavior. | Medium — the dispatch seam makes it tractable |
| Pipelining (vRAM residency) | Every API call is a full upload-compute-download round trip. Composites (Closure, Gaussian, LoG) already ping-pong on-device within one call, but nothing survives across calls. | Medium-far |
| Additional filters / FFT | 19 compute commands today; zero FFT references; global ops (watershed, connected components) absent. Full candidate list now lives in [BACKLOG.md](BACKLOG.md). | Near for compositions, far for global ops |
| C# + NuGet | Nothing. No `extern "C"` API (only mangled C++ templates over `ImageView<T>`), shared-lib target commented out, empty `.def` files. | Far — needs C ABI + shared lib (later phase) |
| Non-NVIDIA GPU | Raw CUDA everywhere; zero OpenCL/SYCL/HIP references. The only backend seam is the generated `CudaCall_<Name>::run` stubs calling CUDA drivers directly. | Far — deliberately deferred (later phase) |
| GPU-direct disk I/O | The C++ core has zero file I/O by design (in-memory `ImageView` only). No GDS/nvImageCodec/nvTIFF references. | Far — and of questionable value (later phase) |

Cross-cutting problems the path below fixes along the way:

- **Version chaos**: `pyproject.toml`, `recipe/meta.yaml`, and `vcpkg.json` say 0.1.0; the MATLAB `.prj` says 3.1.2;
  git tags reach v3.15; CMake declares no version.
  Decision: continue the 3.x lineage — the packaging-era relaunch is **v4.0.0**.
- **CI has never executed a filter**: all runners are GPU-less GitHub-hosted machines, so accuracy tests always skip.
  The CPU backend (Step 3) fixes this permanently.
- **Test data weight**: `test_data/` is ~435 MB of Git-LFS TIFFs — a CI bandwidth hazard as jobs multiply.

## Development workflow

- **`develop` is the integration branch.** It starts by fast-forwarding to the current working HEAD plus this roadmap
  (`origin/develop` is a clean ancestor — 0 unique commits, 31 behind — so this is conflict-free).
  The old `conda-forge` dev branch is retired after the fast-forward; that name is reserved for the future feedstock.
- **All work happens on feature branches PR'd into `develop`** (e.g. `feature/version-single-source`,
  `feature/scikit-build-core`, `feature/cpu-backend-infra`). Useful material is pulled or cherry-picked from
  existing branches — notably `feature/vcpkg` — rather than merged wholesale (see Step 0).
- **`main` is the release branch**: untouched until the first `v4.0.0` release PR from `develop`; tags live there.
- **Parallel lanes.** Steps 1-2 (packaging: CMake install/export, recipes, CI) and Step 3 (CPU backend: `src/c/Cpu/`,
  dispatch macro, tests) touch mostly disjoint files and can proceed concurrently on separate feature branches
  once Step 0 lands. Step 4 should follow Step 3, because the CPU parity tests are the safety net for its refactoring.
  Backlog items (Step 5) unlock as Step 3's op waves land.

```text
Step 0: foundations (branch reset, version single-source, recipe fixes, install/export)
    │
    ├──► Step 1: conda / mamba / pixi / uv(sdist) ──► Step 2: vcpkg / conan
    │        (CPU wheels + conda CPU variant are Step 3 deliverables that close Step 1 follow-ups)
    │
    └──► Step 3: CPU fallback (parallel lane with Steps 1-2)
                │
                └──► Step 4: pipelining (design doc ──► device-apply refactor ──► executor)

Step 5: operation backlog (BACKLOG.md — seeded now, groomed continuously)
Later:  C ABI ──► C# + NuGet;  non-NVIDIA GPU assessment;  GPU-direct disk I/O
```

Effort key (solo-maintainer scale): **S** = days, **M** = 1-3 weeks, **L** = 1-2 months, **XL** = a quarter or more.

## Step 0 — Foundations (~2-3 weeks; prerequisite for everything)

All items are S unless noted.

- **Branch reset**: fast-forward `develop` to the current working HEAD, commit this roadmap and `BACKLOG.md` there,
  retire the `conda-forge` branch name. `main` waits for the v4.0.0 release.
- **Salvage `feature/vcpkg` by file, not by merge** (M).
  It diverged from the same commit as the current work and both sides modified the CMakeLists files, so a merge is all conflicts.
  Take verbatim: `cmake/hydra-config.cmake.in`; `src/c/Version.h.in` (adapt `@GITVERSION_*@` placeholders to `@PROJECT_VERSION*@`);
  `ports/hydra/vcpkg.json`, `ports/hydra/portfile.cmake`, and `scripts/update-vcpkg-files.py` (parked until Step 2).
  Re-implement by hand on a feature branch: the install/export blocks — `GNUInstallDirs`,
  `configure_package_config_file` + `write_basic_package_version_file`, `install(EXPORT HydraTargets NAMESPACE Hydra::)`,
  and headers installed to `include/hydra`.
  Skip `.gitversion.yml` and the self-hosted `hydra-ci.yml` (documented later as an option).
- **Single-source the version**: a plain-text `VERSION` file at repo root containing `4.0.0`.
  Git-derived schemes (GitVersion, setuptools_scm) fail exactly where packaging needs them:
  conda-forge and vcpkg build from GitHub tarballs that contain no `.git` directory.
  Consumers: CMake reads it into `project(VERSION)` and configures `Version.h.in`;
  `pyproject.toml` via scikit-build-core's regex metadata provider (Step 1); `meta.yaml` via Jinja `load_file_regex`;
  the MATLAB `.prj` is patched by the release workflow.
  A CI guard asserts git tag == `VERSION` on every tag push.
- **Fix the known recipe bugs**: add the missing backslash continuations in `recipe/build.sh`
  (today only the first line of the cmake invocation runs on Linux);
  replace the phantom `python -m unittest src/Python/Test/test_accuracy.py` test with import checks;
  fix `about.home`/`dev_url` (they point at `zfphil`, not `ericwait`).
- **License/metadata cleanup**: keep `LICENSE.md` as canonical, delete the duplicate `license.txt`,
  pick one author email everywhere, and add build-from-source instructions to `readme.md`
  (required for registry reviews anyway).
- **Small code prep for Step 3**: fix the `sum_array`/`sum` binding bug in `src/Python/hydra_image_processor/core.py`
  (the fallback path raises `AttributeError` instead of the intended `NotImplementedError`);
  reserve `HYDRA_DEVICE_CPU = -2` in `src/c/Cuda/Defines.h` (`-1` already means "all GPUs");
  add a `HYDRA_HOST_DEVICE` macro (`__host__ __device__` under `__CUDACC__`, empty otherwise).
- **Make the CUDA arch list a cache variable** (`HYDRA_CUDA_ARCHITECTURES`):
  mainline pins `75;86;89;120` while `feature/vcpkg` changed it to `89;90;103;121` —
  keep the mainline list as default and let packagers override.

## Step 1 — conda / mamba / pixi / uv, with a documented update process

**Goal**: a user can `conda install`/`mamba install`/`pixi add` a GPU build, `uv pip install` an sdist,
and the maintainer can publish an update with one tag push.

- **Python build integration — scikit-build-core** (L). The prerequisite for everything Python-facing.
  Move `pyproject.toml` to the repo root and let `pip install .` / `uv pip install .` drive the top-level CMake.
  Add `install(TARGETS HydraPy ...)` and stop writing build products into the source tree
  (today `src/c/Python/CMakeLists.txt` drops `Hydra.pyd/.so` into `src/Python/hydra_image_processor/`
  and `MANIFEST.in` bundles whatever is lying around —
  a `pip install .` without a prior manual CMake build yields an importable-but-nonfunctional package).
  Standardize on `find_package(Python COMPONENTS Interpreter Development.Module NumPy)`
  (scikit-build-core hints the `Python` find module, not `Python3`).
- **PyPI / uv story, sequenced honestly**: publish the **sdist** now — `uv pip install hydra-image-processor` works
  but compiles from source and requires the CUDA toolkit locally; document this loudly.
  **No CUDA wheels on PyPI** (GB-class, ABI-fragile, a maintenance treadmill).
  CPU-only wheels via cibuildwheel are an explicit **Step 3 deliverable** that upgrades uv users to binary installs.
  Until then, the documented binary path is conda/mamba/pixi.
- **conda-forge feedstock** (M + review latency).
  Changes vs the current recipe: GitHub release tarball URL + `sha256` instead of `source: path: ..`
  (needs the first v4.0.0 release); `{{ compiler('cuda') }}` / `{{ compiler('cxx') }}` / `{{ stdlib("c") }}` conventions;
  imports-only tests for the GPU variant (conda-forge CI has no GPUs — building CUDA packages there is standard).
  Submit linux-64 first, win-64 as a follow-up.
  The CPU variant (`cuda_compiler_version: [None, 12.x]` matrix) is a **Step 3 deliverable**.
  Interim: keep the anaconda.org channel healthy (fix CI trigger branches, add ubuntu to the matrix once `build.sh` is fixed).
  mamba and pixi consume conda-forge automatically; add a root `pixi.toml` with dev/test tasks as contributor convenience (S).
- **Release automation + documentation** (M): one `release.yml` on `v*` tags —
  guard (tag == `VERSION`), GitHub Release with notes, conda build + anaconda.org upload
  (until the feedstock's bot takes over), sdist to PyPI via trusted publishing,
  `.mltbx` build with `.prj` version patched from `VERSION`.
  Write **`RELEASING.md`**: the exact update process (bump `VERSION`, update changelog, tag, what automation does,
  what to verify afterward, how conda-forge/vcpkg pick up the release). "Easy to update" is a deliverable, not a hope.
- **LFS bandwidth**: default `lfs: false` in checkouts; one dedicated data-test job restoring LFS objects
  from `actions/cache`; longer term, move test data to a GitHub Release asset or Zenodo.

## Step 2 — vcpkg / conan

**Goal**: a C++ consumer gets `find_package(Hydra)` from a registry, and each release updates the port automatically.

- **vcpkg** (M). Finish the port parked in Step 0:
  replace `SHA512 0` and `REF v${GITVERSION_SEMVER}` with literals written by `scripts/update-vcpkg-files.py`,
  wired into `release.yml` so every tag opens a port-update PR (extend `RELEASING.md` accordingly).
  Add a `HYDRA_BUILD_BINDINGS=OFF` guard so the port build needs no Python/MATLAB.
  Ship as an **overlay port / small self-hosted registry first**;
  submit to the central microsoft/vcpkg registry after 2-3 stable releases
  (central CI review of CUDA ports is slow, and every release needs a version-database PR).
- **conan** (S-M). Assess after vcpkg works: the Step 0 install/export is package-manager-agnostic,
  so a `conanfile.py` is mostly metadata. Publish to ConanCenter only if users ask;
  otherwise document consuming via `conan` from a git URL.
- **C ABI design note** (S, design only — implementation is a later phase):
  while the export surface is being shaped here, write down the flat C API design (see "Later phases")
  so Step 2's installed headers and targets don't need rework when it lands.

## Step 3 — CPU fallback (parallel lane with Steps 1-2; XL total)

**Goal**: every filter runs on machines without an NVIDIA GPU; implicit fallback **warns**;
a **strict flag** turns fallback into an error; CI finally executes filters.

- **Where the code lives**: new `src/c/Cpu/` mirroring the CUDA headers one-to-one (`CpuMultiplySum.h`, `CpuMaxFilter.h`, ...).
  Each defines a host driver with the **same name and signature** as its CUDA counterpart, inside `namespace CpuBackend`.
  Identical signatures keep the dispatch change to a two-line macro edit.
- **Dispatch, two layers**:
    1. CMake option `HYDRA_CPU_ONLY`: builds `src/c/Cpu/` only — no `LANGUAGES CUDA`, no `find_package(CUDAToolkit)`.
       This unlocks the Step 1 follow-ups (CPU wheels, conda CPU variant) and GPU-less CI.
    2. In dual builds, the generated stub body (`_SCR_GEN_TYPED_IMPL` in `src/c/ScriptCmds/GenCommands.h`, ~lines 182-186)
       becomes: if `device == HYDRA_DEVICE_CPU` or `deviceCount() == 0`, call `CpuBackend::cFoo(...)`,
       else call the CUDA driver.
       Fallback semantics move out of the Python/MATLAB wrappers and into C++, where every frontend inherits them.
       A future HIP backend is another namespace and another branch — no rewrite.
- **Fallback UX (explicit requirements)**:
    - **Warning on implicit fallback**: when `device == -1` (default) and no GPU is found, the dispatch emits a warning
      through each frontend's native channel — Python `RuntimeWarning`, MATLAB `warning('Hydra:cpuFallback', ...)`,
      C/C++ a stderr message (later a registerable callback). Explicitly requesting `device = HYDRA_DEVICE_CPU` is silent —
      intentional CPU use is not a fallback.
    - **Strict mode**: `HYDRA_REQUIRE_GPU=1` environment variable (following the existing `HYDRA_ENABLE_MUTEX` pattern
      in `src/c/ScriptCmds/HydraConfig.h`, surfaced via the `CheckConfig` command) makes implicit fallback an error
      instead of a warning. Useful for benchmarking and for pipelines where silent CPU execution would be misleading.
- **Boundary-condition parity is non-negotiable — share the code, don't reimplement it**:
    - Move `KernelIterator` method bodies into the header and decorate with `HYDRA_HOST_DEVICE`.
      The class is pure `Vec` arithmetic with no CUDA intrinsics,
      so a CPU loop using the same iterator performs **bit-identical per-pixel arithmetic** for neighborhood ops.
      Bitwise equality — not tolerance matching — becomes the test target.
    - Factor the fractional-coordinate trilinear accessor (`CudaImageContainer.cuh`, ~lines 46-100, hit by even-sized kernels)
      into a shared `HYDRA_HOST_DEVICE` free function (e.g. `src/c/Cuda/ImageSample.h`) used by both backends.
      This is exactly where silent divergence would hide.
- **Parallelization**: OpenMP `parallel for` over a flattened output-pixel index
  (mirroring `GetThreadBlockCoordinate` on the GPU side).
  Skip the `ImageChunk` machinery on CPU — host RAM is the ceiling.
  Caveat: MSVC ships OpenMP 2.0, so use signed flattened index loops (or `/openmp:llvm`).
- **Op implementation waves** (each op is S once the infrastructure exists):
    1. MultiplySum (the archetype — validates the whole parity story), Sum, GetMinMax, ElementWiseDifference, IdentityFilter.
    2. MeanFilter, Gaussian (separable, reuses MultiplySum machinery), MinFilter, MaxFilter, MedianFilter.
    3. Closure, Opener, LoG, HighPassFilter — pure compositions of waves 1-2.
    4. StdFilter, VarFilter, EntropyFilter, WienerFilter.
    5. NLMeans (largest single kernel, M).
- **Retire the language-native fallbacks** once the C++ backend covers them:
  delete the Python `local/core.py` stubs and the `_gpu_with_fallback` wrapper;
  deprecate MATLAB `+HIP/+Local` for one release with a warning
  (the toolbox functions pad/reflect at borders and therefore diverge from Hydra's boundary at edge pixels), then remove.
- **Testing and CI payoff** (M, woven through the waves):
    - New `src/c/test_back/test_cpu_parity.cpp` (pattern cloned from `test_accuracy.cpp`):
      CPU vs the `test_data/` ground-truth TIFFs on GitHub-hosted runners,
      plus CPU-vs-GPU bitwise comparison (via `device=-2` vs default) where a GPU exists.
    - Consolidate the loose Python scripts into a pytest suite under `src/Python/tests/`, parameterized over the 8 pixel types.
    - New CPU-only CI workflow (linux/macos/windows matrix, `-DHYDRA_CPU_ONLY=ON`) —
      the first CI in the project's history that computes anything.
    - Close the Step 1 follow-ups: CPU wheels via cibuildwheel to PyPI (uv users get binaries),
      CPU variant added to the conda-forge feedstock.
- **Risk mitigations**: land the `GenCommands.h` macro edit with the CUDA path only and diff preprocessor output
  before adding the CPU branch (an X-macro mistake breaks all 24 commands at once);
  add a dedicated even-kernel interpolation parity test;
  while in there, consider consolidating the 8-pixel-type list —
  currently duplicated across `ScriptCommandImpl.h`, `GenCommands.h`, and `LinkageTraitTfms.h` — into one X-macro header.
  Document clearly that CPU mode is a correctness/accessibility path, not a performance claim.

## Step 4 — Filter pipelining (XL; after Step 3; starts with a design doc)

**Goal**: run a chain of filters keeping data in vRAM as long as possible,
with the ability to emit intermediate results to host memory at chosen points.

This step explicitly begins with a **design document** (`docs/design/pipeline.md`, reviewed as its own PR)
before implementation — the descriptor format, memory model, and frontend API deserve deliberate design.
Direction of record, to be refined by that document:

- **A pipeline API ("declare the chain, execute once"), not exposed device handles.**
  An opaque GPU-array handle crossing the Python/MATLAB boundary drags in cross-language lifetime management,
  multi-GPU placement, and error-state surface —
  and is fundamentally incompatible with chunking (a chunked image cannot stay resident).
  The pipeline generalizes what Closure/Gaussian/LoG already do internally:
  per chunk, upload once, run every op in the chain via `CudaDeviceImages` ping-pong, download once.
  It works whether or not the image fits in vRAM — residency handles do not.
- **Intermediate outputs (host taps)**: each stage descriptor carries an `emit` flag;
  the executor retrieves that stage's ROI to a caller-provided host buffer in addition to feeding the next stage.
  Tapped stages cost one extra device-to-host copy — data still never makes a round trip back up.
  This satisfies "send intermediate results out to host memory" without breaking residency for the rest of the chain.
- **Prerequisite refactor — device-apply split** (L, mechanical, protected by Step 3's parity tests):
  split each CUDA driver into `deviceApply(residentBuffers, params, chunk)` plus the existing transfer shell.
  No behavior change; it is the enabling refactor for the executor and improves code health regardless.
- **Backend**: a `PipelineExecutor` consuming a list of op descriptors and calling the `deviceApply` entry points.
  The chunk halo becomes the **sum** of per-op kernel halos across the chain (extend `ImageChunk.cpp`),
  so interior pixels never see window clipping mid-chain
  and boundary semantics are identical to running the ops as separate calls.
  The CPU backend runs the same descriptor list trivially (taps are just pointer copies), keeping parity testable.
- **Frontend**: one new `SCR_CMD_NOPROC` command (e.g. `RunPipeline`)
  taking the descriptor list through the existing struct/deferred-type machinery,
  plus thin fluent builders in Python and MATLAB.
- Later, if profiling shows transfer-bound single-op workloads on fits-in-vRAM images,
  a residency handle can be added *on top of* the executor — do not lead with it.
- **Open questions for the design doc**: descriptor schema and intermediate pixel-type promotion (the `OutMap` machinery);
  halo-sum growth for long chains of large kernels (mid-chain re-tiling is research-grade — likely document the limit);
  multi-GPU chunk striping interaction with taps (output ordering); error semantics mid-chain;
  how strict mode (`HYDRA_REQUIRE_GPU`) applies to whole-pipeline fallback.

## Step 5 — Operation backlog (seeded now; groomed continuously)

The explicit, priority-ordered list of operations to implement lives in **[BACKLOG.md](BACKLOG.md)** —
one row per operation with priority, effort, chunking compatibility, dependencies, and status.
Reordering rows is reprioritizing; rows are mirrored to GitHub issues as they are picked up.
It includes the FFT work (frequency-domain convolution, bandpass/DoG, Richardson-Lucy deconvolution),
the cheap morphological compositions, new local kernels,
and the **global operations (connected components, watershed, distance transform)** —
which are listed with an explicit architecture caveat:
they require global propagation passes that conflict with the streaming-chunk design,
so each needs either a fits-on-one-GPU gate or a border-merge design before implementation.

Key technical notes that constrain backlog items:

- **FFT stack**: cuFFT on GPU (ships with the toolkit); **pocketfft** on CPU (BSD-3, header-only, powers NumPy/SciPy —
  FFTW rejected on GPL license). Chunking via overlap-add/overlap-save fits the existing `ImageChunk` margin model;
  FFT plan and scratch memory must join the chunk-size budget.
  Boundary story: default to **normalized convolution** (divide the zero-padded convolution by the convolution
  of a ones-mask), which reproduces the energy-insulated boundary exactly for one extra cacheable transform.
  cuFFT is a large *dynamic* redistributable — adding it breaks the "cudart_static-only, small packages" assumption
  that the conda and (future) NuGet packaging rely on; check before landing.
- **Global ops**: correct chunked versions are research-grade border-merging problems.
  The supported near-term posture is documented interop — Hydra for filtering, scikit-image/ITK for global
  segmentation — unless/until a fits-on-one-GPU mode is added.

## Later phases (after the five steps; kept on the map, no dates)

- **C ABI + shared library** (L; design written during Step 2, implemented here):
  `src/c/CApi/` with `HydraC.h` (pure C header), one exported function per operation (~24 total),
  pixel type as an enum in a caller-owned descriptor struct, 8-way type dispatch inside the shim,
  generated from the same `GenCommands.h` X-macros so new ops get C entry points automatically.
  Error codes + thread-local `hydra_last_error_message()`; nothing throws across the ABI.
  Built as `hydra_c` SHARED, statically linking the backend, `CXX_VISIBILITY_PRESET hidden` —
  the templated C++ surface never crosses a DLL boundary. Serves C#, Rust, Julia, Java, and ctypes alike.
  Testable without a GPU thanks to Step 3.
- **C# + NuGet** (M + M, after the C ABI): `bindings/csharp/` netstandard2.0 P/Invoke layer over `HydraC.h`
  with an ergonomic `Span<T>` API; NuGet with `runtimes/{win-x64,linux-x64}/native` layout from release artifacts;
  `dotnet test` runs on the CPU backend.
- **Non-NVIDIA GPU support** (assessment of record): **hipify/ROCm over SYCL/Kokkos** when the time comes.
  The kernels are idiomatic CUDA with direct HIP equivalents, and the Step 3 namespace seam accommodates
  a `HipBackend::` branch without redesign. SYCL/Kokkos would mean rewriting the kernel layer for a heavy dependency —
  unjustified while the CPU backend covers non-NVIDIA users.
  Trigger to act: demonstrated user demand plus access to AMD test hardware. Effort if triggered: L-XL.
- **GPU-direct disk I/O** (recommendation: defer): nvTIFF doesn't cover OME-TIFF/BigTIFF hyperstacks;
  GPUDirect Storage needs `nvidia-fs`/DGX-class deployments (Linux-only); nvImageCodec targets JPEG-family codecs.
  The pragmatic 80% win is **pinned-host-memory double-buffered prefetch** (M) feeding the Step 4 executor
  (e.g. `tifffile`/`zarr` readers filling `cudaHostRegister`-ed buffers), overlapping disk I/O with compute.
  Any I/O layer belongs in the Python package or a `hydra-io` companion — never in `src/c/Cuda`.
  Revisit only after Step 4 ships and profiling shows I/O-bound pipelines.

## Top risks

1. **LFS bandwidth exhaustion** — 435 MB of test data multiplied by a growing CI matrix; address in Step 1 before adding jobs.
2. **conda-forge review latency for CUDA recipes** (weeks) — the anaconda.org channel is the hedge;
   the Step 3 CPU variant strengthens the submission.
3. **X-macro edits break all 24 commands at once** — mitigate with staged landing and preprocessor-output diffing (Step 3).
4. **cuFFT vs the "cudart_static-only, small redistributables" assumption** that conda and NuGet packaging rely on —
   a standing cross-check whenever FFT backlog items start.
5. **Version-drift regression** — the tag == `VERSION` CI guard is the enforcement;
   without it the five-version chaos returns.
6. **CPU performance expectations** — document CPU mode as a correctness and accessibility path, not a performance claim.
