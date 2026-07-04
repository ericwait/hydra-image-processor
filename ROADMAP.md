# Hydra Image Processor Roadmap

This document assesses how far the current implementation is from the project's ideals and lays out a dependency-ordered path to get there.
It was last grounded against the codebase in July 2026 (branch `conda-forge`, commit `c7f1f07`); file paths and claims reference that state.

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
| vcpkg / conan | Unmerged `feature/vcpkg` branch has near-complete CMake install/export and a port skeleton with a `SHA512 0` placeholder. Mainline has zero `install()` rules, no `project(VERSION)`. | Medium — salvage and finish |
| conda / mamba / pixi | Working-ish recipe with Windows-only CI uploading to an anaconda.org channel. Not on conda-forge. Known bugs listed in Phase 0. | Medium |
| C# + NuGet | Nothing. No `extern "C"` API (only mangled C++ templates over `ImageView<T>`), shared-lib target commented out, empty `.def` files. | Far — needs C ABI + shared lib |
| Non-NVIDIA GPU | Raw CUDA everywhere; zero OpenCL/SYCL/HIP references. The only backend seam is the generated `CudaCall_<Name>::run` stubs calling CUDA drivers directly. | Far — deliberately deferred (Phase 2) |
| CPU fallback | C++: none — the build fails without the CUDA toolkit. Python: 100% `NotImplementedError` stubs. MATLAB: ~14/19 real toolbox fallbacks, but with divergent boundary behavior. | Medium — the dispatch seam makes it tractable |
| Additional filters | 19 compute commands today. Several new filters are nearly free as compositions of existing ones. | Near for tier 1 |
| FFT filters | Zero FFT references anywhere. All convolution is direct spatial; Gaussian is separable spatial. | Medium |
| Pipelining (vRAM residency) | Every API call is a full upload-compute-download round trip. Composites (Closure, Gaussian, LoG) already ping-pong on-device within one call, but nothing survives across calls. | Medium-far |
| GPU-direct disk I/O | The C++ core has zero file I/O by design (in-memory `ImageView` only). No GDS/nvImageCodec/nvTIFF references. | Far — and of questionable value (Phase 5) |

Cross-cutting problems the path below fixes along the way:

- **Version chaos**: `pyproject.toml`, `recipe/meta.yaml`, and `vcpkg.json` say 0.1.0; the MATLAB `.prj` says 3.1.2;
  git tags reach v3.15; CMake declares no version.
  Decision: continue the 3.x lineage — the packaging-era relaunch is **v4.0.0**.
- **CI has never executed a filter**: all runners are GPU-less GitHub-hosted machines, so accuracy tests always skip.
  The CPU backend fixes this permanently.
- **Stale `main`**: all recent work lives on the `conda-forge` dev branch; `origin/main` is 22 commits behind with nothing unique.
- **Test data weight**: `test_data/` is ~435 MB of Git-LFS TIFFs — a CI bandwidth hazard as jobs multiply.

## The path

Effort key (solo-maintainer scale): **S** = days, **M** = 1-3 weeks, **L** = 1-2 months, **XL** = a quarter or more.

```text
Phase 0 (foundations, ~2-3 weeks)
    ├──► Track A: CPU backend ─────────────► v4.0.0 release
    ├──► Track B: packaging/install ───────►    │
    │                                            ├──► Phase 2: C ABI + shared lib ──► C# + NuGet
    │                                            ├──► Phase 2: device-apply refactor ──► Phase 3: pipelining ──► FFT deconvolution
    │                                            ├──► Phase 4: FFT convolution/bandpass
    │                                            └──► Phase 5: GPU-direct I/O (assess-only, last)
    Ongoing: filter catalog (tier 1 unlocks right after Track A starts landing ops)
```

### Phase 0 — Foundations (~2-3 weeks total)

Everything else builds on these. All items are S unless noted.

- **Consolidate branches**: merge the `conda-forge` dev branch into `main` (main has nothing unique) and retire the branch name —
  reserve `conda-forge` for the future feedstock.
- **Salvage `feature/vcpkg` by file, not by merge** (M).
  Both branches diverged from the same commit and both modified the CMakeLists files, so a merge is all conflicts.
  Take verbatim: `cmake/hydra-config.cmake.in`; `src/c/Version.h.in` (adapt `@GITVERSION_*@` placeholders to `@PROJECT_VERSION*@`);
  `ports/hydra/vcpkg.json`, `ports/hydra/portfile.cmake`, and `scripts/update-vcpkg-files.py` (parked until Phase 1B).
  Re-implement by hand on mainline: the install/export blocks — `GNUInstallDirs`,
  `configure_package_config_file` + `write_basic_package_version_file`, `install(EXPORT HydraTargets NAMESPACE Hydra::)`,
  and headers installed to `include/hydra`.
  Skip `.gitversion.yml` and the self-hosted `hydra-ci.yml` (documented later as an option).
- **Single-source the version**: a plain-text `VERSION` file at repo root containing `4.0.0`.
  Git-derived schemes (GitVersion, setuptools_scm) fail exactly where packaging needs them:
  conda-forge and vcpkg build from GitHub tarballs that contain no `.git` directory.
  Consumers: CMake reads it into `project(VERSION)` and configures `Version.h.in`;
  `pyproject.toml` via scikit-build-core's regex metadata provider (Phase 1B); `meta.yaml` via Jinja `load_file_regex`;
  the MATLAB `.prj` is patched by the release workflow.
  A CI guard asserts git tag == `VERSION` on every tag push.
- **Fix the known recipe bugs**: add the missing backslash continuations in `recipe/build.sh`
  (today only the first line of the cmake invocation runs on Linux);
  replace the phantom `python -m unittest src/Python/Test/test_accuracy.py` test with import checks;
  fix `about.home`/`dev_url` (they point at `zfphil`, not `ericwait`).
- **License/metadata cleanup**: keep `LICENSE.md` as canonical, delete the duplicate `license.txt`,
  pick one author email everywhere, and add build-from-source instructions to `readme.md`
  (required for registry reviews anyway).
- **Small code prep for Track A**: fix the `sum_array`/`sum` binding bug in `src/Python/hydra_image_processor/core.py`
  (the fallback path raises `AttributeError` instead of the intended `NotImplementedError`);
  reserve `HYDRA_DEVICE_CPU = -2` in `src/c/Cuda/Defines.h` (`-1` already means "all GPUs");
  add a `HYDRA_HOST_DEVICE` macro (`__host__ __device__` under `__CUDACC__`, empty otherwise).
- **Make the CUDA arch list a cache variable** (`HYDRA_CUDA_ARCHITECTURES`):
  mainline pins `75;86;89;120` while `feature/vcpkg` changed it to `89;90;103;121` —
  keep the mainline list as default and let packagers override.

### Phase 1, Track A — CPU backend (near-term centerpiece; XL total; ships in v4.0.0)

**Why first**: it fixes "build requires nvcc," gives all three frontends automatic fallback in one place,
makes the library usable on non-GPU machines, lets CI actually execute filters for the first time,
makes the conda-forge feedstock testable, and is the honest non-NVIDIA story until a second GPU backend is justified.

- **Where the code lives**: new `src/c/Cpu/` mirroring the CUDA headers one-to-one (`CpuMultiplySum.h`, `CpuMaxFilter.h`, ...).
  Each defines a host driver with the **same name and signature** as its CUDA counterpart, inside `namespace CpuBackend`.
  Identical signatures keep the dispatch change to a two-line macro edit.
- **Dispatch, two layers**:
    1. CMake option `HYDRA_CPU_ONLY`: builds `src/c/Cpu/` only — no `LANGUAGES CUDA`, no `find_package(CUDAToolkit)`.
       This is the packaging enabler.
    2. In dual builds, the generated stub body (`_SCR_GEN_TYPED_IMPL` in `src/c/ScriptCmds/GenCommands.h`, ~lines 182-186) becomes:
       if `device == HYDRA_DEVICE_CPU` or `deviceCount() == 0`, call `CpuBackend::cFoo(...)`, else call the CUDA driver.
       Fallback semantics move out of the Python/MATLAB wrappers and into C++, where every frontend inherits them.
       A future HIP backend is another namespace and another branch — no rewrite.
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
- **Risk mitigations**: land the `GenCommands.h` macro edit with the CUDA path only and diff preprocessor output
  before adding the CPU branch (an X-macro mistake breaks all 24 commands at once);
  add a dedicated even-kernel interpolation parity test;
  while in there, consider consolidating the 8-pixel-type list —
  currently duplicated across `ScriptCommandImpl.h`, `GenCommands.h`, and `LinkageTraitTfms.h` — into one X-macro header.
  Document clearly that CPU mode is a correctness/accessibility path, not a performance claim.

### Phase 1, Track B — Packaging and install (parallel with Track A)

- **Python build integration — scikit-build-core** (L).
  Move `pyproject.toml` to the repo root and let `pip install .` drive the top-level CMake.
  Add `install(TARGETS HydraPy ...)` and stop writing build products into the source tree
  (today `src/c/Python/CMakeLists.txt` drops `Hydra.pyd/.so` into `src/Python/hydra_image_processor/`
  and `MANIFEST.in` bundles whatever is lying around —
  a `pip install .` without a prior manual CMake build yields an importable-but-nonfunctional package).
  Standardize on `find_package(Python COMPONENTS Interpreter Development.Module NumPy)`
  (scikit-build-core hints the `Python` find module, not `Python3`).
  Once `HYDRA_CPU_ONLY` exists, auto-select it when no CUDA toolkit is found so `pip install .` works everywhere.
- **Wheel strategy**: **no CUDA wheels on PyPI.**
  They are GB-class, ABI-fragile, and a maintenance treadmill for a solo maintainer.
  Ship: sdist always; CPU-only wheels via cibuildwheel once Track A lands
  (small, testable on GPU-less CI, gives every `pip install hydra-image-processor` a working baseline);
  GPU binaries via conda. Document "GPU users: conda/mamba/pixi" prominently.
- **conda-forge feedstock** (M + review latency).
  Requirements vs the current recipe: GitHub release tarball URL + `sha256` instead of `source: path: ..`
  (needs the first v4.0.0 release); `{{ compiler('cuda') }}` / `{{ compiler('cxx') }}` / `{{ stdlib("c") }}` conventions;
  imports-only tests for the GPU variant (conda-forge CI has no GPUs — building CUDA packages there is standard).
  Submit linux-64 first, win-64 as a follow-up.
  When Track A lands, add the CPU variant (`cuda_compiler_version: [None, 12.x]` matrix) —
  that variant runs real filter tests on conda-forge CI.
  Interim: keep the anaconda.org channel healthy (fix CI trigger branches, add ubuntu to the matrix once `build.sh` is fixed).
  mamba and pixi consume conda-forge automatically;
  optionally add a root `pixi.toml` with dev tasks as contributor convenience (S).
- **vcpkg** (M). Finish the parked port:
  replace `SHA512 0` and `REF v${GITVERSION_SEMVER}` with literals written by `scripts/update-vcpkg-files.py`,
  wired into the release workflow so every tag refreshes the port.
  Add a `HYDRA_BUILD_BINDINGS=OFF` guard so the port build needs no Python/MATLAB.
  Ship as an **overlay port / small self-hosted registry first**;
  submit to the central microsoft/vcpkg registry after 2-3 stable releases
  (central CI review of CUDA ports is slow, and every release needs a version-database PR).
- **Conan**: skip for now, vcpkg-first.
  The Phase 0 install/export work is package-manager-agnostic, so a conanfile is cheap later if users ask.
- **Release automation** (M): one `release.yml` on `v*` tags —
  guard (tag == `VERSION`), GitHub Release with notes,
  conda build + anaconda.org upload (until the feedstock's bot takes over),
  vcpkg port-update PR, `.mltbx` build with `.prj` version patched from `VERSION`;
  later phases append sdist/CPU wheels (PyPI trusted publishing), C-ABI shared libs, and NuGet push.
- **LFS bandwidth**: default `lfs: false` in checkouts;
  one dedicated data-test job restoring LFS objects from `actions/cache`;
  longer term, move test data to a GitHub Release asset or Zenodo.

### Phase 2 — Bridge: C ABI, shared library, backend seam hardening (post-4.0)

- **C ABI + shared library** (L; design it early — other bindings benefit from targeting it):
    - New `src/c/CApi/`: `HydraC.h` (pure C public header), `HydraCTypes.h`, `HydraC.cpp`.
    - Shape: **one exported C function per operation** (~19 + ~5 utilities),
      with pixel type carried as an enum inside a caller-owned image descriptor struct
      (`dims[3]`, `channels`, `frames`, `pixel_type`, `data`) mirroring `ImageView`/`ImageDimensions`.
      The 8-way type dispatch happens inside the shim,
      generated from the same `GenCommands.h` X-macros so new ops get C entry points automatically.
      (Rejected: per-op-per-type exports — ~150 brittle symbols;
      a single string-dispatch entry — discards compile-time signature checking and needs a third arg-converter stack.)
    - Errors: every function returns an `int32_t` status; `hydra_last_error_message()` with thread-local storage;
      every call wrapped in `try/catch` — nothing throws across the ABI.
    - Build: `hydra_c` as a SHARED library statically linking `HydraCudaStatic`,
      `CXX_VISIBILITY_PRESET hidden` so only the `extern "C"` surface exports.
      This sidesteps the C++ ABI tar-pit entirely — the templated surface never crosses a DLL boundary.
      `CUDA_RESOLVE_DEVICE_SYMBOLS` and PIC are already set on the static target;
      verify device-symbol resolution on both platforms. Delete the empty `.def` files.
    - Build it against the CPU backend too, so the C ABI is testable in GPU-less CI.
      This layer also serves Rust/Julia/Java/ctypes users, not just C#.
- **C# + NuGet** (M + M, after the C ABI):
    - `bindings/csharp/`: netstandard2.0 class library,
      `[DllImport("hydra_c")]` declarations mirroring `HydraC.h` (~24 functions — hand-written is fine),
      an ergonomic `Span<T>`-based layer, and tests running on the CPU backend so `dotnet test` needs no GPU.
    - NuGet layout: managed lib in `lib/netstandard2.0/`,
      natives in `runtimes/win-x64/native/` and `runtimes/linux-x64/native/`, pulled from release artifacts.
    - CUDA redistribution: `cudart_static` is currently the only CUDA dependency, so packages stay small.
      **Standing cross-check**: the FFT phase adding cuFFT (a large dynamic redistributable) would break this —
      if that happens, split a CPU-only base package from a GPU add-on package.
- **Device-apply refactor** (L, mechanical, fully covered by the parity tests):
  split each CUDA driver into `deviceApply(residentBuffers, params, chunk)` plus the existing transfer shell.
  No behavior change; it is the prerequisite for pipelining and improves code health regardless.
- **Non-NVIDIA GPU assessment** (assessment only, no dates): **hipify/ROCm over SYCL/Kokkos** when the time comes.
  The kernels are idiomatic CUDA (`__constant__` kernel memory, `cudaMemcpyToSymbol`, occupancy API)
  with direct HIP equivalents, so `hipify` keeps the code in-dialect,
  and the Track-A namespace seam accommodates a `HipBackend::` branch without redesign.
  SYCL/Kokkos would mean rewriting the entire kernel layer and adopting a heavy dependency —
  unjustified for a solo-maintained library whose non-GPU users already have the CPU backend.
  Trigger to act: demonstrated user demand plus access to AMD test hardware. Effort if triggered: L-XL.

### Phase 3 — Filter pipelining (XL; depends on the device-apply refactor)

**Recommendation: a pipeline API ("declare the chain, execute once"), not exposed device handles.**

- Rationale: an opaque GPU-array handle crossing the Python/MATLAB boundary drags in cross-language lifetime management,
  multi-GPU placement, and error-state surface —
  and is fundamentally incompatible with chunking (a chunked image cannot stay resident).
  The pipeline generalizes what Closure/Gaussian/LoG already do internally:
  per chunk, upload once, run every op in the chain via `CudaDeviceImages` ping-pong, download once.
  It works whether or not the image fits in vRAM — residency handles do not.
- Backend: a `PipelineExecutor` consuming a list of op descriptors and calling the Phase-2 `deviceApply` entry points.
  The chunk halo becomes the **sum** of per-op kernel halos across the chain (extend `ImageChunk.cpp`),
  so interior pixels never see window clipping mid-chain
  and boundary semantics are identical to running the ops as separate calls.
  The CPU backend runs the same descriptor list trivially, keeping parity testable.
- Frontend: one new `SCR_CMD_NOPROC` command (e.g. `RunPipeline`)
  taking the descriptor list through the existing struct/deferred-type machinery,
  plus thin fluent builders in Python and MATLAB.
- Later, if profiling shows transfer-bound single-op workloads on fits-in-vRAM images,
  a residency handle can be added *on top of* the executor — do not lead with it.
- Risks: halo-sum shrinks usable chunk size for long chains of large kernels
  (document; mid-chain re-tiling is research-grade — defer);
  descriptors must carry intermediate pixel-type promotion (the `OutMap` machinery).

### Phase 4 — FFT filters (L for convolution/bandpass; + L for deconvolution, which wants Phase 3)

- Ops that earn their keep: frequency-domain convolution for large kernels (direct cost explodes above roughly 15^3),
  difference-of-Gaussians/bandpass, and **Richardson-Lucy deconvolution** —
  the marquee microscopy feature, and iterative (convolve, divide, convolve, multiply per iteration),
  which is exactly the shape the pipeline executor accelerates. Sequence R-L after Phase 3.
- Libraries: cuFFT on GPU (ships with the toolkit — no new dependency, but see the NuGet size cross-check above).
  CPU: **pocketfft** (BSD-3, header-only, powers NumPy/SciPy).
  FFTW is rejected on license (GPL); kissfft is BSD but slower.
- Chunking: overlap-add/overlap-save with kernel-sized padding per chunk — fits the existing `ImageChunk` margin model.
  FFT plan and scratch memory must join the chunk-size budget calculation.
- Boundary story (be explicit in docs): circular convolution has no native equivalent of clipped-window renormalization.
  Default to **normalized convolution** — divide the zero-padded convolution by the convolution of a ones-mask —
  which reproduces the energy-insulated boundary exactly for one extra cacheable transform.
  Offer plain pad modes as documented alternatives.
  Parity tests against spatial MultiplySum validate the path
  (tolerance-based, not bitwise — cuFFT and pocketfft round differently).

### Phase 5 — GPU-direct disk I/O (assess-only; recommendation: defer)

Honest assessment of the NVIDIA decoder/storage stack against this project's users
(1-5D microscopy volumes on lab workstations):

- nvTIFF covers baseline TIFF and a subset of compressions —
  OME-TIFF/BigTIFF multi-page hyperstacks with varied bit depths routinely exceed it.
- GPUDirect Storage (cuFile) requires supported filesystems and the `nvidia-fs` driver stack —
  DGX-class deployments, not typical lab machines, and Linux-only.
- nvImageCodec targets JPEG-family codecs, not microscopy formats.

The pragmatic 80% win needs none of that: **pinned-host-memory double-buffered prefetch** (M) —
e.g. Python-side `tifffile`/`zarr` readers filling `cudaHostRegister`-ed buffers that feed the Phase-3 pipeline executor,
overlapping disk I/O with compute using only existing dependencies.
If an I/O layer is ever built, it belongs in the Python package or a `hydra-io` companion —
never in `src/c/Cuda` (the core's zero-I/O design is a feature).
Revisit only after Phase 3 ships and profiling shows I/O-bound pipelines.

### Ongoing — filter catalog (prioritized for microscopy)

Current inventory (19 compute commands): Closure, ElementWiseDifference, EntropyFilter, Gaussian, GetMinMax,
HighPassFilter, IdentityFilter, LoG, MaxFilter, MeanFilter, MedianFilter, MinFilter, MultiplySum, NLMeans,
Opener, StdFilter, Sum, VarFilter, WienerFilter (plus 5 meta-commands).

- **Tier 1 — nearly free compositions, chunking-compatible (S each, any time after Track A's waves land)**:
  top-hat/bottom-hat (Opener/Closure + ElementWiseDifference), morphological gradient (Max − Min),
  unsharp mask (HighPassFilter exists), difference-of-Gaussians.
- **Tier 2 — new local kernels, chunking-compatible (S-M each)**:
  Niblack/Sauvola adaptive thresholding (Mean and Std already exist), Sobel/gradient magnitude, bilateral filter,
  anisotropic diffusion (iterative-local — a natural pipeline citizen),
  Otsu global threshold (a histogram is a mergeable reduction like GetMinMax, so it is chunk-safe despite being "global-valued").
- **Tier 3 — fundamentally global: architecture warning.**
  Watershed, connected components, distance transform, and morphological reconstruction
  require global propagation passes that **conflict with the streaming-chunk architecture**;
  correct chunked versions are research-grade border-merging problems.
  Declare them out of scope for the chunked engine (or gate behind an explicit fits-on-one-GPU mode),
  and document the intended workflow: Hydra for filtering, scikit-image/ITK for global segmentation steps.
  Do not let these creep in without that decision.

## Top risks

1. **LFS bandwidth exhaustion** — 435 MB of test data multiplied by a growing CI matrix; address in Track B before adding jobs.
2. **conda-forge review latency for CUDA recipes** (weeks) — the anaconda.org channel is the hedge;
   the CPU variant strengthens the submission.
3. **X-macro edits break all 24 commands at once** — mitigate with staged landing and preprocessor-output diffing (Track A).
4. **cuFFT vs the "cudart_static-only, small redistributables" assumption** that the conda and NuGet packaging rely on —
   a standing cross-track check when Phase 4 starts.
5. **Version-drift regression** — the tag == `VERSION` CI guard is the enforcement;
   without it the five-version chaos returns.
6. **CPU performance expectations** — document CPU mode as a correctness and accessibility path, not a performance claim.
