# StereoForge

Geometry-aware stereo video synthesis, starting with **one continuous, uncut
recording** such as drone footage or a walkthrough.

The implemented pipeline is:

```text
Video → ORB/RANSAC keyframes → VGGT-Ω dense geometry → ALIKED/LightGlue tracks
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

The script builds the CUDA → base → geometry → stereo → pyCuSFM image chain with
BuildKit and starts an interactive container. If the named container is already
running, it attaches without rebuilding. Exit/stop it before rebuilding changed
Dockerfiles or native code. Python source and configuration changes are visible
through their explicit workspace mounts.

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

Inside Docker, with your continuous video under `data/input/`:

```bash
python -m stereoforge.geometry.demo --video data/input/forest_road.mp4
```

For a naturally ordered image folder:

```bash
python -m stereoforge.geometry.demo --images data/input/frames
```

Video inputs now use visual keyframes by default. VGGT distributes overlapping sections across
visible GPUs with adaptive memory budgets, then unloads its models before CPU
merging. ALIKED refinement follows. Set `refinement.enabled: false` in
`configs/default.yaml` for VGGT only. ALIKED is the only supported refinement feature family.

The native selector extracts ORB features in bounded parallel CPU workers, then
makes decisions in timestamp order against the **last accepted keyframe**. Mutual
Hamming ratio matching is verified with fundamental-matrix and homography RANSAC.
Inlier support, spatial coverage, sharpness, overlap retention and image movement
control selection. A recent reliable frame can bridge a sudden overlap loss;
unresolved tracking breaks stop geometry rather than silently joining disconnected
views. RANSAC provides geometric filtering, not a guarantee that every match is correct.
ORB is only the selection frontend; pyCuSFM continues to use ALIKED.

To inspect keyframe selection alone, rebuild the Docker environment after native
code changes, then run inside the container with the desktop display forwarded:

```bash
python -m stereoforge.video.keyframe_demo --video data/input/barn.mp4
```

To compare **SuperPoint + LightGlue + RANSAC**, use:

```bash
python -m stereoforge.video.keyframe_demo --video data/input/barn.mp4 \
    --keyframe-config configs/keyframes_superpoint.json
```

After inspecting selection, the same configuration works end to end:

```bash
python -m stereoforge.geometry.demo --video data/input/barn.mp4 \
    --keyframe-config configs/keyframes_superpoint.json
```

Rebuild Docker for the native backend and export dependencies. First use downloads
the SuperPoint and matching LightGlue weights, exports separate ONNX models, and
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
allocation. Inputs and outputs have fixed dimensions; adaptive depth is disabled. Export follows upstream's Dynamo/opset-20 settings, including
its integer-division translation for TensorRT. A compatibility pass folds constant
shape expressions and materializes ONNX initializers for reduction axes required
by TensorRT 10.13. Both ONNX files are checked before
cache publication or engine building. The export contract version invalidates
older graphs; no manual cache deletion is required.


The learned backend uses one SuperPoint worker per visible GPU when GPU 0 can
access its peers, cached reference features, and ordered LightGlue matching on
GPU 0. Without peer access it uses GPU 0 for both stages. It starts with mixed FP16, 1024
keypoints, and a 960 × 544 model canvas. Images retain their aspect ratio and are
padded on the bottom/right; padding and weak detections cannot become RANSAC
matches. `model_width`/`model_height` control that canvas; `feature_edge` and
`descriptor_ratio` apply only to ORB. `detector_threshold` filters SuperPoint
detections and `match_threshold` filters LightGlue confidence. Selection thresholds
in the learned config are initial values, **not yet calibrated or benchmarked**.
ORB remains the default and pyCuSFM still uses ALIKED.

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

Floating-point SuperPoint outputs bind directly to separate LightGlue reference
and candidate inputs on the same GPU. Descriptors never round-trip through host
memory. For another extraction GPU, a peer copy creates a retained GPU-0 mirror
once per frame; this is **not** zero-copy across GPUs. TensorRT still performs its
internal pair assembly/normalization as part of the LightGlue graph.

Cached PNGs are decoded on the CPU and uploaded once through pinned memory. CUDA
area resizing, normalization, and padding write directly to SuperPoint's input.
A shared-memory Laplacian stencil and warp reductions compute sharpness on GPU;
small keypoint/score/statistic and match arrays return to CPU for RANSAC/debugging.
This removes intermediate tensor transfers, but is **not** a fully device-only
NVDEC-to-SuperPoint video path: the existing decoded-PNG cache boundary remains.
The area resize and FP16 sharpness path can slightly change threshold decisions.


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

Tune the initial thresholds in `configs/keyframes.json`, or supply another file
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
- `video/` and `native/video/`: Python process adapter and C++ FFmpeg extraction.
- `utils/`: shared camera conversions, progress, artifacts and visualization.

Future stages belong in the architecture roadmap until implemented. Package
initializers describe real packages; empty helper scripts, test files and future
modules are deliberately absent. Builds, tests and inference remain user-run in
the current development workflow.
