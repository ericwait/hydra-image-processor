# Operation Backlog

The re-arrangeable, priority-ordered list of filters and operations to implement (Step 5 of [ROADMAP.md](ROADMAP.md)).

How to use this file:

- **Reordering rows within a priority band — or moving a row between bands — is reprioritizing.** Edit freely; PR to `develop`.
- When work starts on a row, open a GitHub issue, link it in the Status column, and move status to `in progress`.
- After Step 3 (CPU backend) lands, every new operation ships **both** backends (`Cuda*.cuh` + `Cpu*.h`)
  plus a parity test against ground truth in `test_data/` — that is part of the definition of done.
- New operations follow the three-file pattern in [CLAUDE.md](CLAUDE.md): CUDA/CPU driver, `ScrCmd*.h`, one `SCR_CMD` line.

Column notes — **Effort**: S = days, M = 1-3 weeks, L = 1-2 months (per backend where two exist).
**Chunk-safe**: whether the op fits the streaming-chunk architecture (local neighborhood or mergeable reduction).

## Priority 1 — cheap wins (compositions of existing ops)

| Operation | Kind | Effort | Chunk-safe | Depends on | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Top-hat | Composition | S | yes | Opener, ElementWiseDifference | proposed | `image - open(image)`; bright feature extraction |
| Bottom-hat | Composition | S | yes | Closure, ElementWiseDifference | proposed | `close(image) - image`; dark feature extraction |
| Morphological gradient | Composition | S | yes | MaxFilter, MinFilter | proposed | `max(image) - min(image)`; edge strength |
| Difference of Gaussians | Composition | S | yes | Gaussian | proposed | Two sigmas, subtract; blob enhancement |
| Unsharp mask | Composition | S | yes | HighPassFilter | proposed | HighPass already exists; add amount parameter |

## Priority 2 — new local kernels (chunking-compatible)

| Operation | Kind | Effort | Chunk-safe | Depends on | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Sobel / gradient magnitude | New kernel | S | yes | — | proposed | Separable derivative kernels |
| Niblack / Sauvola threshold | New kernel | S-M | yes | MeanFilter, StdFilter | proposed | Local adaptive thresholding; Mean/Std already exist |
| Otsu threshold | Reduction | S-M | yes | histogram reduction | proposed | Histogram is a mergeable reduction (like GetMinMax), so chunk-safe despite being global-valued |
| Bilateral filter | New kernel | M | yes | — | proposed | Edge-preserving smoothing; range+spatial weights |
| Anisotropic diffusion | Iterative kernel | M-L | yes | — | proposed | Perona-Malik; halo cost per iteration; natural Step 4 pipeline citizen |

## Priority 3 — FFT track

Shared prerequisites for this band: cuFFT (GPU) + vendored pocketfft (CPU, BSD-3); overlap-add/save chunking;
normalized-convolution boundary mode (see ROADMAP Step 5 notes).
Standing check: cuFFT is a large dynamic redistributable — coordinate with conda/NuGet packaging before landing.

| Operation | Kind | Effort | Chunk-safe | Depends on | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| FFT convolution (large kernels) | FFT | L | yes (overlap-add) | FFT infrastructure | proposed | Wins over spatial above ~15^3 kernels; parity-test vs MultiplySum |
| FFT bandpass / DoG | FFT | S after FFT conv | yes | FFT convolution | proposed | Frequency-domain band selection |
| Richardson-Lucy deconvolution | FFT, iterative | L | yes | FFT convolution; Step 4 pipeline (strongly) | proposed | Marquee microscopy feature; iterative conv/divide/multiply loops |
| Wiener deconvolution | FFT | M | yes | FFT convolution | proposed | Non-iterative alternative to R-L; distinct from existing spatial WienerFilter |

## Priority 4 — global operations (architecture caveat)

These require global propagation/label passes that **conflict with the streaming-chunk architecture**.
Each row needs a design decision before implementation: either a **fits-on-one-GPU gate**
(error on images that would chunk) or a research-grade **border-merge design**.
Until then, the documented workflow is Hydra for filtering, scikit-image/ITK for global segmentation.

| Operation | Kind | Effort | Chunk-safe | Depends on | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Connected components | Global label | L | no — needs gate or border-merge | design decision | proposed | Union-find on GPU is well-studied for single-buffer images |
| Distance transform | Global sweep | L | no — needs gate or border-merge | design decision | proposed | Prerequisite for watershed seeding |
| Watershed | Global propagation | L-XL | no — needs gate or border-merge | connected components, distance transform | proposed | Most-requested segmentation op; hardest to chunk correctly |
| Morphological reconstruction | Global iteration | L | no — needs gate or border-merge | design decision | proposed | Enables h-maxima/h-minima, hole filling |

## Priority 5 — ideas / unscheduled

| Operation | Kind | Effort | Chunk-safe | Depends on | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Hole filling | Composition of reconstruction | S after reconstruction | no | morphological reconstruction | proposed | |
| H-maxima / H-minima | Composition of reconstruction | S after reconstruction | no | morphological reconstruction | proposed | Seed detection for watershed |
| Local entropy-based threshold | New kernel | M | yes | EntropyFilter | proposed | |
| Structure tensor / orientation | New kernel | M | yes | Sobel | proposed | Fiber/orientation analysis |
