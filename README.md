# StereoForge

Geometry-aware stereo video synthesis, starting with **one continuous, uncut
recording** such as drone footage or a walkthrough.

The default reconstruction pipeline grows one sparse map:

```text
Video → keyframes → verified RaCo–ALIKED/LightGlue+ tracks
      → small VGGT-Ω seed → initial pyCuSFM BA
      → PnP registration of remaining keyframes → triangulation + periodic BA
      → full-sequence colored sparse viewer
```

Start a new end-to-end run inside Docker:

```bash
python -m stereoforge.reconstruction.demo --video data/input/barn.mp4
```

The default seed is the first 32 selected keyframes (`--seed-frames 16–64`).
Only the seed receives VGGT poses. All later poses are estimated from 2D–3D
correspondences against the growing map. There is no required section Sim(3).
The same registration code is shared with the successful bounded experiment.
A camera passing the training PnP checks enters a temporary map for joint BA.
The unchanged held-out checks then decide whether to commit it. Failed proposals
roll back the camera, added landmarks and all BA changes; other candidates can proceed. Missing frames
are retried after map growth; a pass with no progress produces an explicit partial
result. Completion requires every selected keyframe, including all seed cameras.

Open `data/intermediate/incremental_TIMESTAMP/index.html`; `report.json` lists
missing frames, PnP attempts and validation before/after candidate BA. `map_ba/` retains native logs and
COLMAP models. `incremental_state.json` atomically points to the last BA checkpoint
that preserved supported cameras and passed withheld-pixel checks. Resume with:

```bash
python -m stereoforge.reconstruction.demo --resume data/intermediate/incremental_TIMESTAMP
```

Resume reuses frames, tracks, the VGGT seed and the last validated map. Uncommitted
registrations are recomputed after interruption. Rejected native BA artifacts are
retained but never committed. Input/checkpoint identity and track/policy identity
are checked. The root viewer covers the whole requested sequence; a partial map
is never labeled complete. Sparse reconstruction is not dense surface reconstruction.

Initial calibration for later frames is the median optimized seed intrinsics.
A training-only refinement can adjust pose and a common fx/fy multiplier within
±10%, keeping the principal point and aspect ratio fixed. Bounds, conditioning,
and training scores determine whether to retain this adjustment; unchanged held-out
checks decide camera acceptance after joint BA. Held-out pixels never enter that BA. Per-attempt focal diagnostics are in `report.json`.
This assumes constant image dimensions and is not a general self-calibration solution.
Existing incremental checkpoints retain their map and held-out observation identities
when resumed under this registration policy. PnP and triangulation run on CPU, TensorRT matching and
pyCuSFM BA use the selected GPU. One VGGT seed uses one GPU and unloads before BA.
No nonlocal loop retrieval or dense-depth refinement is added by this path.
End-to-end runtime validation remains user-run.

Legacy `hierarchical_*` resumes retain the old section-merging behavior; they are
not silently converted. Use `--video` for the new incremental pipeline.

The existing dense geometry pipeline remains available for comparison:

```text
Video → RaCo–ALIKED/LightGlue+ + CUDA RANSAC keyframes → VGGT-Ω dense geometry → ALIKED/LightGlue tracks
      → VGGT depth-initialized pyCuSFM bundle adjustment → WebGL reports
```

Baseline control, StereoSpace inference and side-by-side video encoding are the
next stages; the application does not yet generate stereo video. See the
[StereoForge Architecture Specification](StereoForge%20Architecture%20Specification.md)
for module responsibilities, camera conventions, limitations and the roadmap.

## Environment

Initialize the pinned upstream repositories when setting up a checkout:

```bash
git submodule update --init --recursive
bash scripts/build_and_start.sh
```

The script builds the CUDA → base → geometry → stereo → cuNLS image chain with
BuildKit and starts an interactive container. If the named container is already
running, it attaches without rebuilding. Exit/stop it before rebuilding changed
Dockerfiles or native code. Python source and configuration changes are visible
through their explicit workspace mounts.

The final image builds the unchanged `third_party/cuNLS` checkout against CUDA 13.2.
It installs C++ headers/library and CMake targets under `/opt/cunls`, the matching
cuDSS shared libraries under `/opt/cudss/lib`, and `pycunls` plus CUDA 13 CuPy in
our existing venv. CuPy stays below v14 to preserve NumPy 1.26. Upstream pycunls
metadata still names CUDA 12 CuPy; its dependencies are installed explicitly, so
package dependency checkers may report that metadata mismatch. The source is not patched.
`find_package(cunls CONFIG REQUIRED)` exposes `cunls::cunls` for our future native solver.

**Backend migration in progress:** pyCuSFM is no longer installed. The existing
reconstruction/refinement commands still call it and will not run in this image
until the custom cuNLS backend is implemented. Video decoding and keyframe selection
remain available; installing cuNLS does not yet migrate reconstruction.

The environment uses the NVIDIA runtime, all visible GPUs, privileged mode,
host network/PID/IPC, and X11 mounts. It mounts source, configuration, data
and cache individually. It does not mount the entire repository.

All Python packages share `/opt/stereoforge-venv`. Dependencies are installed with
uv, except for the existing TensorRT pip installation, and constrained by
`docker/constraints.txt`. There is no `uv sync` workflow or dependency lockfile.
Native C++20 video extraction and its CUDA resize kernel are built in the geometry image. Dependency or native
changes require an image rebuild; no empty packaging manifest is maintained.

## Checkpoint access

Obtain access to [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega). On the
host, put your Hugging Face read token in `.secrets/hf_token` as a single plain
line. Prepare the private file using:

```bash
mkdir -p .secrets
chmod 700 .secrets
touch .secrets/hf_token
chmod 600 .secrets/hf_token
```

Edit the file locally, then recreate the container so the startup script mounts
it read-only at `/run/secrets/hf_token`. It sets `HF_TOKEN_PATH`; the token value
is not placed in build arguments or environment variables. Credentials, data
and caches are ignored by Git and excluded from the Docker build context.

The demo uses `vggt_omega_1b_512.pt` from the persistent Hugging Face cache and
downloads it from the gated model repository if missing. Obtain model access
approval and use a token belonging to that approved account for the initial
download. Cached checkpoints are reused without a fresh authorization request.
No download flag or interactive login is needed. `--checkpoint PATH` remains an
explicit override for a local checkpoint you are authorized to use; it does not
perform a Hugging Face access check.

## Run the geometry pipeline

The former hierarchical approach is retained only for resuming its saved runs.
Its section-alignment experiments do not run in new incremental reconstructions.

The earlier dense pipeline is invoked separately below.

Inside Docker, with your continuous video under `data/input/`:

```bash
python -m stereoforge.geometry.demo --video data/input/forest_road.mp4
```

For a naturally ordered image folder:

```bash
python -m stereoforge.geometry.demo --images data/input/frames
```

Video inputs now use visual keyframes by default. VGGT distributes overlapping sections across
visible GPUs with adaptive memory budgets, then unloads its models before
merging. ALIKED refinement follows. Set `refinement.enabled: false` in
`configs/default.yaml` for VGGT only. ALIKED is the only supported refinement feature family.

Section alignment defaults to 32 shared keyframes (`geometry.chunk_overlap`).
It uses camera/depth seeds and deterministic Sim(3) RANSAC with unchanged quality
checks. A rejected merge retains all section tensors and `failure.json` under
`data/intermediate/failed_reconstructions/<id>/` for inspection. These tensors
include processed RGB and geometry; their original staging image paths may no
longer exist. Retention does not automatically resume a failed run.

The merger verifies exact shared-frame RGB identity and requires a majority
consensus among camera/depth-derived transforms. It then attempts robust,
multiscale point-to-plane ICP on overlap surfaces with scale held fixed.
Original pixel-correspondence and camera checks decide acceptance, not ICP fitness.
Planar or otherwise underconstrained ICP updates are discarded. Diagnostics record
per-frame transforms, pairwise agreement, and ICP results on success and failure.
These thresholds remain heuristic and need validation on representative footage.

ICP uses Open3D's tensor API: voxel downsampling, normals, and registration run
on CUDA when available. Geometry preparation and final acceptance checks remain
on CPU. Each overlap uses one GPU; it does not require two GPUs. Automatic device selection warns before falling back to CPU if Open3D CUDA
is unavailable. Diagnostics record the actual ICP device. No CPU retry hides CUDA runtime errors.

The dense merger registers adjacent sections only through identical shared
frames. It verifies RGB identity, fits Sim(3) and ICP on training overlap images,
and validates on held-out overlap images. Rejected boundaries retain section
tensors under `data/intermediate/failed_reconstructions/` without publishing a merge.
The obsolete standalone alignment/reprojection experiments have been removed.

The native selector uses RaCo–ALIKED + LightGlue+ in FP16 TensorRT by default,
with CUDA fundamental-matrix and homography RANSAC. Feature extraction runs in
bounded GPU workers; decisions remain in timestamp order against the **last
accepted keyframe**. RaCo–ALIKED/LightGlue+ is the only keyframe frontend.
Inlier support, spatial coverage, sharpness, overlap retention and image movement
control selection. A recent reliable frame can bridge a sudden overlap loss;
unresolved tracking breaks stop geometry rather than silently joining disconnected
views. RANSAC provides geometric filtering, not a guarantee that every match is correct.
The keyframe frontend is separate from pyCuSFM, which continues to use ALIKED.

To inspect keyframe selection alone, rebuild the Docker environment after native
code changes, then run inside the container with the desktop display forwarded:

```bash
python -m stereoforge.video.keyframe_demo --video data/input/barn.mp4
```

To select **RaCo–ALIKED + LightGlue+ + CUDA RANSAC** explicitly, use:

```bash
python -m stereoforge.video.keyframe_demo --video data/input/barn.mp4 \
    --keyframe-config configs/keyframes_raco.json
```

After inspecting selection, the same configuration works end to end:

```bash
python -m stereoforge.geometry.demo --video data/input/barn.mp4 \
    --keyframe-config configs/keyframes_raco.json
```

Rebuild Docker for the native backend and export dependencies. First use downloads
the RaCo, ALIKED and matching LightGlue+ weights, exports separate ONNX models, and
builds local TensorRT engines. This preparation can take several minutes; progress
reports the current export/build stage. Engines persist under
`.cache/stereoforge/keyframes/`, keyed by model settings, adapter, GPU identity,
and runtime versions. The existing TensorRT 10.13.3.9 version is retained.

Prepare models separately before opening the debugger (inside Docker):

```bash
python -m stereoforge.video.learned_models
```

The supplied config selects `"precision": "fp16"`. This builds mixed-FP16 TensorRT
engines for each visible GPU, using FP16 floating-point I/O and FP32 sensitive
arithmetic. Match indices remain INT32 so all configured keypoint indices are exact. It is
reduced-precision optimization, not calibrated INT8 quantization. Accuracy and
speed still need comparison against FP32 on real frames.

```bash
# Download missing weights and export/cache only ONNX; no GPU is needed.
python -m stereoforge.video.learned_models --onnx-only

# Build an FP32 reference from the same cached ONNX models.
python -m stereoforge.video.learned_models --precision fp32
```

Set `precision` in the demo's config to the precision you want it to use. A CLI
build override does not modify that config. ONNX files have a separate cache
under `.cache/stereoforge/keyframes/onnx/`; precision and GPU changes reuse them.
Both ONNX and engine caches use hashes, locks, and atomic publication. Engine-build
failures preserve the completed ONNX cache. Our separate model interfaces differ
from the upstream combined ONNX release, so preparation downloads upstream weights
and exports compatible ONNX files once rather than loading an incompatible graph.

LightGlue-ONNX is pinned as the `third_party/lightglue-onnx` Git submodule and
copied into Docker. Initialize it with `git submodule update --init --recursive`
when updating an existing checkout. The adapter uses upstream attention and
assignment modules, then applies mutual-match filtering directly into fixed-size
buffers. It avoids the upstream variable-length match list and its NonZero
allocation. Inputs and outputs have fixed dimensions; adaptive depth is disabled.
RaCo uses dense keypoint ranking and folded BatchNorm. ALIKED uses upstream's
portable GridSample decomposition of deformable convolution, avoiding a separate
TensorRT plugin dependency. This is an FP16-compatible deployment path, not a
claim of matching upstream's fastest benchmark configuration. Export follows upstream's Dynamo/opset-20 settings, including
its integer-division translation for TensorRT. A compatibility pass folds constant
shape expressions and materializes ONNX initializers for reduction axes required
by TensorRT 10.13. Both ONNX files are checked before
cache publication or engine building. The export contract version invalidates
older graphs; no manual cache deletion is required.


The learned backend uses one RaCo–ALIKED worker per visible GPU when GPU 0 can
access its peers, cached reference features, and ordered LightGlue matching on
GPU 0. Without peer access it uses GPU 0 for both stages. It starts with mixed FP16, 1024
keypoints, and a 960 × 544 model canvas. Images retain their aspect ratio and are
padded on the bottom/right; padding and detections below the configured cutoff
cannot become RANSAC matches. `model_width`/`model_height` control that canvas.
`detector_threshold` filters RaCo spatial detection probabilities (default zero),
and `match_threshold` filters LightGlue confidence. Selection thresholds
in the learned config are initial values, **not yet calibrated or benchmarked**.
RaCo is the only keyframe frontend; pyCuSFM still uses ALIKED.

The TensorRT build uses optimization level 5, VRAM-based memory limits, available
cuBLAS/cuBLASLt/cuDNN/edge-mask/JIT tactics, existing-weight sparsity, and the full
runtime. Before each engine build, the selected GPU's free and total VRAM are
queried. The tactic budget is free VRAM minus a reserve of 10% of total VRAM
(at least 1 GiB); operation workspace is capped at half that budget to leave room
for weights, activations and build overhead. Both limits are rounded down to
powers of two for TensorRT's pool validation and read back before building to
check that TensorRT accepted them. These are pool limits, not a hard
process-wide memory cap or an upfront allocation. Each engine's `.build.json`
records the chosen byte limits. Builder objects are released before engine
inspection. Shared-memory tactic limits stay at the GPU's hardware maximum. It does
not force sparsity into dense weights or force normalization/softmax into FP16.
The C++ runner uses spin waits and caches up to 32 CUDA graphs by tensor addresses;
unsupported capture falls back to normal enqueue. Reusable frame slots keep GPU
allocations and addresses stable across candidates. No speedup is claimed before
profiling on the target machine.

Floating-point RaCo–ALIKED outputs bind directly to separate LightGlue reference
and candidate inputs on the same GPU. Descriptors never round-trip through host
memory. For another extraction GPU, a peer copy creates a retained GPU-0 mirror
once per frame; this is **not** zero-copy across GPUs. TensorRT still performs its
internal pair assembly/normalization as part of the LightGlue graph.

Cached PNGs are decoded as BGR on the CPU and uploaded once through pinned memory.
CUDA converts them to RGB NCHW using antialiased bicubic resizing (Keys cubic,
a=-0.5), half-pixel centers and clamped source edges. Separable horizontal and
vertical passes widen filter support for downsampling and normalize weights at
each output pixel. The horizontal pass writes a reusable FP32 GPU workspace;
the vertical pass clamps cubic overshoot to [0,1] and writes the FP16/FP32 model
input. The canvas is zero-padded on the bottom/right. Both passes use 32×8 blocks
and bounded shared-memory strips; no image data returns to the CPU between them.
A shared-memory Laplacian stencil and warp reductions compute sharpness on GPU;
RANSAC reads TensorRT keypoints, scores, indices and confidence on the same CUDA
stream, without a host round trip. It samples 2,000 hypotheses per model family,
uses normalized DLT (rank-two enforcement for fundamental matrices), rejects
rank-deficient samples, scores consensus in parallel and refits on winning inliers.
Coverage and median displacement are computed on-device. Only aggregate statistics
return to the CPU during production selection; the debug window additionally
copies keypoints and verified matches for display. The CUDA eight-point solver
is not numerically identical to OpenCV's CPU estimator and requires validation.
This removes intermediate tensor transfers, but is **not** a fully device-only
NVDEC-to-RaCo video path: the existing decoded-PNG cache boundary remains.
Bicubic resizing and FP16 sharpness can change threshold decisions. Rebuilding
the native selector invalidates its cached selections; TensorRT engines remain
reusable. Interpolation accuracy and performance still require validation.


The standalone commands open an OpenCV window, initially paused. **N** steps to the next candidate,
**Space** plays/pauses, **O** toggles rejected matches, and **Q/Esc** quits and saves
a partial report. The left image is the reference used for the decision; the right
is the candidate. Green lines are RANSAC inliers; red lines (optional) are rejected
descriptor matches. Captions show the decision, feature count, sharpness, inlier
support, coverage, and motion. Playback is processing-paced, not video-rate.
If selection inserts a bridge keyframe, the displayed reference is that bridge.
The final usable endpoint may also be accepted when selection finishes.

The standalone debugger does not run VGGT or pyCuSFM. Decoding reuses its cache,
but selection always runs again.
Results go under `data/intermediate/keyframes_*/selection.json`; candidate images
are retained there as hard links when possible. `interrupted`, `tracking_break`,
and `insufficient_keyframes` are diagnostic statuses, not successful reconstructions.
Use `--input PATH_TO_DECODED_FRAMES` instead of `--video` to reuse a directory with
its native `manifest.json` directly. Use all candidate frames, not an already
filtered geometry run, when evaluating selection thresholds.

Tune the initial thresholds in `configs/keyframes_raco.json`, or supply another file
with `--keyframe-config`. These defaults have not been calibrated on Barn. Use
`--all-frames` to bypass selection for diagnostics. `--frames` limits video
candidates **before** keyframe selection, so the resulting keyframe count can be
smaller. Image-folder inputs are treated as already selected and are unchanged.

This first implementation finishes parallel decoding into the persistent cache,
then performs parallel feature extraction and ordered selection. It still caches
all candidate PNGs; only selected images are linked into a run and passed to VGGT,
geometry reports and pyCuSFM. Selection results are cached separately by selector
binary and settings. `input_frames/manifest.json` records acceptance reasons,
per-candidate match metrics, original displayed-frame indices and timestamps.
Failed selections retain their JSON diagnostic in the decoded cache. At least
three connected keyframes are required. The original video is retained; full-rate
stereo rendering will require geometry/poses for intervening frames in a later stage.

Video extraction uses NVIDIA NVDEC when the codec/profile and installed FFmpeg
support it, with CPU fallback. A CUDA area resizer reduces supported decoded
surfaces before copying them to host memory; a bounded pool of up to eight CPU
workers converts RGB and writes PNGs. For full videos, the extractor detects visible
CUDA devices and indexes compressed packet timestamps/keyframes before assigning
independent sections to multiple GPUs. Each worker seeks to its section, decodes
the required preceding frames and saves only its assigned timestamp interval.
Sections decode from up to two earlier keyframes to establish reference frames,
then retain only their assigned interval. Packets marked by FFmpeg as decode-only
are excluded from the expected display-frame index, but still reach the decoder.
The combined result must match that timestamp index exactly before publication.
One progress bar counts saved frames across devices. The writer budget is shared
across devices, with at least one writer per section. Unsupported indexing or
unreliable section boundaries fall back to sequential extraction; explicit frame/time
selections use the sequential path. Multiple GPUs do not guarantee proportional
speedups: indexing and disk writes still take time.
If validation falls back to a sequential pass, its timestamp mismatch and packet
counts are retained as `input_frames/parallel_diagnostic.json` in the completed run.
Sequential fallback still attempts NVIDIA acceleration.

Geometry working images have a maximum edge
of three times the configured VGGT resolution (1536 pixels for the default model).
Images are never enlarged or cropped during extraction; dimensions round down to
even pixels for chroma compatibility. VGGT still applies its own final preprocessing
and predicts intrinsics for those processed dimensions. This changes pixel sampling
compared with full-resolution extraction; the original video remains untouched.

Completed extractions are cached beside the video in `data/input/.frames/` for
the usual input location. Rerunning the same source and selection reuses the PNGs,
hard-linking them into the new run instead of duplicating storage where possible.
Treat these images as immutable. Source file metadata, extraction options, and
native executable contents determine cache identity. Incomplete extractions are
never reused. `input_frames/manifest.json` records source/working dimensions,
timestamps, file sizes and the decoder used. Deleting `.frames/` releases cached
files without removing frames already linked into saved runs.

The standalone extractor accepts `--max-edge 0` to retain original resolution
and `--hardware cpu` for decoder diagnostics. The geometry demo chooses its
working resolution automatically. After native changes, stop the existing
container before running `bash scripts/build_and_start.sh`; otherwise the script
attaches to that container instead of rebuilding.

Optional diagnostic flags include `--frames`, `--start-seconds`, `--duration`,
`--device cuda:0`, `--output`, and `--debug`. Frame/time selections are explicit;
they are not required for full-video processing. Only use `--meters-per-unit`
when scale is independently calibrated.

## Inspect results

Open the new host-side `data/intermediate/geometry_<timestamp>/index.html`.
It opens the final colored pyCuSFM reconstruction, or a diagnostic page if
refinement failed. There is one 3D viewer for a refined run, with points, camera
frustums, trajectory and progressive playback. Source depth/confidence previews
are still labeled as VGGT references; dense arrays remain available for recovery.
If refinement is explicitly disabled, the result is a VGGT-only viewer.

Standalone BA exports black point colors. Validation restores the original
RGB-sampled seed colors through checked observation associations, preserving the
native export separately. Missing refined cameras are reported explicitly.
Refinement uses one elapsed-time progress bar whose stage label and counters
change in place; detailed native output remains in stage log files. Geometry
serialization likewise uses one progress bar.

Useful artifacts inside each run:

| Path | Contents |
|---|---|
| `geometry.npz`, `metadata.json`, `run.json` | Dense arrays, camera data and provenance |
| `previews/`, `contact_sheet.jpg` | Source RGB/depth/confidence inspection |
| `pycusfm/initialization.json` | Track/landmark counts and unsupported frames |
| `pycusfm/initialized_sparse/` | Readable depth-initialized COLMAP model |
| `pycusfm/initialized_binary/` | Binary input used to bypass the native text-import bug |
| `pycusfm/workspace/ba_raw/` | Unmodified bundle-adjustment output |
| `pycusfm/workspace/sparse/` | Validated sparse COLMAP model |
| `pycusfm/refinement.json`, `pycusfm/comparison.json` | Refinement settings and coverage/quality statistics |
| `pycusfm/*.log` | Native stage logs |

Refinement failures preserve completed VGGT results and diagnostics. A successful
native process exit does not by itself establish a valid reconstruction. Partial
coverage remains partial; full coverage is not an accuracy guarantee. Dense depth
and refined cameras still need geometric reconciliation before stereo synthesis.

## Recover without repeating the full pipeline

Refine a selection from saved VGGT geometry:

```bash
python -m stereoforge.geometry.refine_demo \
  --run data/intermediate/geometry_<timestamp> \
  --frames 16 --frame-step 16 --debug
```

This creates a new `pycusfm_diagnostic_<timestamp>/` folder. Frame spacing is a
diagnostic tool to cover more camera motion; full-video processing retains all
frames.

Retry only BA from a full run's saved initialized landmarks:

```bash
python -m stereoforge.geometry.ba_demo \
  --run data/intermediate/geometry_<timestamp>
```

If that BA completed but report generation failed, reuse its optimized output:

```bash
python -m stereoforge.geometry.ba_demo \
  --run data/intermediate/geometry_<timestamp> \
  --ba-result data/intermediate/geometry_<timestamp>/pycusfm_ba_<timestamp>
```

Both commands create a new `pycusfm_ba_<timestamp>/` report. Open its `index.html`
directly. Report-only recovery also points the main run entry page at the new
viewer; previous result directories remain intact. `--ba-result` can also name
the full run’s `pycusfm/` directory to refresh its colors without rerunning BA. The validator restores filenames from
stable image IDs after checking observation coordinates and camera associations.

To refresh only the existing VGGT viewer from its saved arrays:

```bash
python -m stereoforge.utils.viewer --run data/intermediate/geometry_<timestamp>
```

Current Dockerfiles include `protobuf>=5,<7` for native match import. Older running
containers can install it with `uv pip install 'protobuf>=5,<7'` before refinement.

## Source organization

- `geometry/`: dense geometry, GPU scheduling, orchestration and existing CLI commands.
- `refinement/`: ALIKED tracks, depth-seeded native BA, COLMAP exchange and sparse reports.
- `stereoforge/video/` and `stereoforge/native/video/`: Python process adapter and C++ FFmpeg extraction.
- `utils/`: shared camera conversions, progress, artifacts and visualization.

Future stages belong in the architecture roadmap until implemented. Package
initializers describe real packages; empty helper scripts, test files and future
modules are deliberately absent. Builds, tests and inference remain user-run in
the current development workflow.

Native file conventions: custom CUDA kernels and their launch implementations use
`.cu` under `stereoforge/native/video/src/`; kernel interfaces use `.cuh` under
`include/`. Host C++ code uses `.cpp`/`.hpp`, including CUDA runtime/driver and
TensorRT callers. `cuda_resize.hpp` is a host class interface; its custom kernels
are implemented in `cuda_resize.cu`. Both language targets use C++20.

## Bounded registration against a saved map

To investigate node 34 without rerunning video processing, matching or VGGT:

```bash
python -m stereoforge.reconstruction.boundary_demo \
  --run data/intermediate/hierarchical_20260917_042617_082322 \
  --node 34 --extension 16 --device 0
```

This separate experiment uses the left child's map before the overlap as its
anchor. It removes overlap cameras and observations, then estimates those cameras
and up to 16 subsequent keyframes from saved global tracks using CPU PnP/RANSAC.
The right child supplies intrinsics only; its poses and 3D points are not imported.
No Sim(3) or five-degree landmark eligibility rule is used.

Registration requires 40 unambiguous 2D–3D correspondences, at least 30 training
inliers, 70% three-pixel support in both training and held-out observations, and
six occupied cells of a 4×4 image grid. Held-out observations stay excluded from
subsequent PnP retries, triangulation and BA. Missing validation landmarks count
as failures. These observations are not independent of the earlier anchor BA.

After each accepted frame, new tracks are triangulated and pyCuSFM refines the
map. The first camera is fixed; the remaining anchor cameras can refine. Loss of
supported cameras or failed held-out checks stops the experiment. At most three
passes are attempted, stopping earlier if a pass registers no cameras. `--extension`
is limited to 0–32 frames; zero tests only the overlap.

Outputs go into a new `boundary_TIMESTAMP/` folder inside the original run:
`report.json` contains registration/rejection counts and per-round BA validation;
`ba_*/` retains native logs and sparse models. Completed or stalled runs also save
`sparse/`, `point_cloud.ply`, and `index.html`. A stalled/failed run returns nonzero.
This does not resume the full hierarchy or establish metric scale. Runtime validation
remains pending; no new feature matching or ACE0 training is performed.
