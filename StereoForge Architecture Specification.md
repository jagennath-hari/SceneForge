# StereoForge Architecture Specification

## Product intent and current scope

StereoForge will convert a monocular video into stereoscopic side-by-side video.
Its central idea is to use scene geometry to choose a virtual stereo baseline
that adapts over time, then synthesize the second eye with StereoSpace.

The current deliverable is **inspectable geometry and sparse refinement for one
continuous, uncut recording**. Drone footage and walkthroughs are the intended
inputs. Edited movies, shot detection, live visualization, baseline control,
StereoSpace inference, and video encoding are outside the implemented scope.
Their intended behavior is described in the roadmap, not represented by empty
Python modules or placeholder tests.

The application currently performs:

```text
Continuous video or ordered image folder
    → decoded RGB frames and timestamps
    → adaptive multi-GPU VGGT-Ω inference
    → CPU alignment and merging
    → saved dense geometry and VGGT viewer
    → ALIKED/LightGlue matching
    → multi-frame tracks initialized from VGGT depth
    → pyCuSFM standalone bundle adjustment
    → validated sparse reconstruction and comparison viewer
```

Refinement is enabled by default and can be disabled in configuration. A failed
refinement must not discard the completed VGGT geometry. A sparse reconstruction
is a separate result; its poses must not silently replace the dense geometry's
camera poses.

## Repository structure

```text
StereoForge/
├── StereoForge Architecture Specification.md
├── README.md                         # Setup, commands, output locations
├── configs/default.yaml             # Active geometry, preview, refinement settings
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.geometry
│   ├── Dockerfile.stereo
│   ├── Dockerfile.pycusfm
│   └── constraints.txt
├── scripts/build_and_start.sh        # Build environment and enter/attach to it
├── native/video/
│   ├── CMakeLists.txt
│   ├── include/stereoforge/video/video_extractor.hpp
│   └── src/                         # FFmpeg RAII owners, extractor, CLI
├── stereoforge/
│   ├── geometry/
│   │   ├── types.py                 # GeometrySequence and FrameGeometry
│   │   ├── config.py                # Validated demo configuration
│   │   ├── inputs.py                # Image discovery and input validation
│   │   ├── vggt_omega.py            # Model adapter and processed-image calibration
│   │   ├── adaptive.py              # GPU scheduling, memory adaptation, CPU spooling
│   │   ├── alignment.py             # Robust overlap Sim(3) alignment
│   │   ├── geometry_utils.py        # Valid-depth masks and confidence statistics
│   │   ├── runner.py                # Full geometry/refinement orchestration
│   │   ├── demo.py                  # Full pipeline CLI
│   │   ├── refine_demo.py           # Refinement from saved VGGT arrays
│   │   └── ba_demo.py               # BA-only and report-only recovery CLI
│   ├── refinement/
│   │   ├── pycusfm.py               # Native stages, working units, cache, process lifecycle
│   │   ├── tracks.py                # Native protobuf reader and unambiguous tracks
│   │   ├── landmarks.py             # Depth initialization and COLMAP text seeds
│   │   ├── colmap_binary.py         # Binary seed serialization and native BA command
│   │   ├── colmap_validation.py     # Output identity and geometric validation
│   │   └── report.py                # Sparse import, alignment, viewer, failure report
│   ├── video/sampling.py            # Python adapter for native frame extraction
│   └── utils/
│       ├── artifacts.py            # Staged publication and JSON artifacts
│       ├── camera.py               # Pose/quaternion convention conversions
│       ├── progress.py             # Progress displays
│       ├── validation.py           # Shared scalar validation
│       ├── visualization.py        # Dense arrays, previews, PLY, HTML
│       ├── viewer.py               # Regenerate the saved VGGT viewer
│       └── templates/geometry.html # Self-contained WebGL viewer
├── third_party/                     # Pinned Git submodules
│   ├── vggt-omega/
│   ├── pycusfm/
│   └── stereospace/
├── data/                            # Local inputs and generated artifacts; ignored
├── .cache/                          # Persistent download/engine cache; ignored
└── .secrets/                        # Optional runtime HF token; ignored
```

Package `__init__.py` files identify real packages. Empty future-stage modules,
empty tests, and inactive scripts are not part of the source tree. Python is run
from the mounted workspace; dependency installation belongs to Dockerfiles and
`docker/constraints.txt`, rather than an empty packaging manifest.

## Environment boundary

The image chain is CUDA → base → geometry → stereo → pyCuSFM. Each Dockerfile has
an explicit named stage and accepts `BASE_FROM`. The current base is
`nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. PyTorch uses the CUDA 13.2 wheel index;
TensorRT is pinned to the version selected for the bundled pyCuSFM CUDA-13 binaries.
This does not mean the closed upstream binaries were rebuilt for CUDA 13.2.

One uv-created virtual environment is shared by all Python dependencies. The
legacy TensorRT installation uses pip in that same environment. BuildKit caches
package downloads. Native video extraction is built inside the geometry image.
StereoSpace remains installed as preparation for the next stage, but is not yet
called by the application. There is no Compose deployment or development-only
Dockerfile.

The build/start script only builds and enters the environment; it does not launch
inference. It selects the NVIDIA runtime, exposes GPUs, and uses the requested
privileged, host-network/PID/IPC and GUI settings. Host IPC is shared memory access,
not a promise of physically unlimited memory. Source, configuration, data and
cache have explicit mounts; the whole repository is not bind-mounted.

The optional `.secrets/hf_token` is mounted read-only at runtime using
`HF_TOKEN_PATH`. Its contents must not enter image layers, build arguments, Git,
or logs. Submodules and their binary/model assets remain upstream-owned.
GTSfM informed the depth-initialization design but is not a runtime dependency.

The default VGGT checkpoint is resolved through the persistent Hugging Face
cache, with an authenticated download from the gated repository when absent.
Initial download requires model access approval for the token's account. Cached
files are reused without an online access check. There is no separate checkpoint
folder or mount. The explicit `--checkpoint PATH` override is reserved for locally
available checkpoints the user is authorized to use; it does not enforce gating.

## Geometry and camera contracts

`FrameGeometry` owns CPU tensors for one processed frame:

| Field | Shape | Meaning |
|---|---|---|
| `depth` | H × W | Camera-forward Z depth |
| `confidence` | H × W | VGGT confidence score, not a probability |
| `intrinsics` | 3 × 3 | Pinhole calibration in processed-image pixels |
| `camera_to_world` | 4 × 4 | Transform from camera coordinates into the sequence world |

`GeometrySequence` also stores processed RGB as N × 3 × H × W, source names,
original image dimensions, and explicit scale metadata. Camera axes are right,
down, forward. All adapters must respect resizing/cropping when associating RGB,
depth, keypoint coordinates and intrinsics. COLMAP poses use world-to-camera
Hamilton quaternions in w,x,y,z order; conversions occur at the integration boundary.

Unknown scale is recorded as `reconstruction_units`. Only an explicit calibrated
meters-per-unit factor allows distances to be labeled meters. That factor scales
both depth and camera translations. Relative similarity alignment is not metric
calibration.

## Video and adaptive inference

The C++20 FFmpeg extractor decodes every frame by default, writes PNGs, flushes
delayed frames at EOF, and records timestamps in a manifest. Its Python adapter
handles process lifecycle and progress; it contains no second decoding loop.
Native code uses namespaces, classes, explicit types, RAII resource ownership,
`[[nodiscard]]` where appropriate, and `this->` for instance member access.

VGGT inference uses the visible CUDA devices, estimating per-device capacity and
adapting overlapping section sizes from memory measurements. The target budget is
90% of each GPU's VRAM, accounting for other allocations; VRAM is not pooled and
90% utilization is not guaranteed. OOM retries reduce section sizes while retaining
all selected frames. Explicit diagnostic selections are the only intentional
frame reduction.

Section results are spooled to CPU/disk. Models are unloaded and CUDA caches
released before CPU merging. Robust Sim(3) fits shared-frame depth geometry,
transforms cameras, and rescales depths. Earlier overlap predictions are retained
once. The first section sets the world/scale anchor. Alignment failure stops the
reconstruction; sequential drift is still possible. Final arrays and compressed
report export require host RAM proportional to sequence length.

## ALIKED and depth-initialized refinement

1. Export processed RGB, per-frame intrinsics, poses and timestamps for pyCuSFM.
   There is one physical monocular sensor; per-frame calibration IDs preserve
   VGGT's changing predicted intrinsics, not a multi-camera rig.
2. Normalize uncalibrated camera translation so the median positive step is one
   working unit. Apply exactly the same transform and scale to depth-derived seeds.
3. Run ALIKED extraction, vocabulary building, pose-graph optimization and
   ALIKED-specific LightGlue matching. Nearby-camera radius search supplies pairs
   without requiring a loop closure. Zero matches stop refinement.
4. Read the pinned native protobuf keypoints and verified pair matches. Build
   union-find tracks; reject components with competing observations in one frame.
   Require at least three usable observations per retained track.
5. Unproject VGGT depth at each observation into the common world frame. Initialize
   each landmark with a confidence-weighted average. Filter observations whose
   initial reprojection error exceeds 14 pixels, then update the seed from retained
   samples. Real matched image coordinates are the BA measurements; projected
   synthetic correspondences are never substituted.
6. Save reciprocal image/point tracks in COLMAP text form for inspection and in a
   separate binary-only directory for native BA. The pinned native text point
   importer misreads numeric RGB fields; binary input bypasses that bug.
7. Run the bundled standalone `bundle_adjustment_runner` with Cauchy loss, first
   camera fixed and re-triangulation disabled. Use its exported optimized cameras
   and points. Its intrinsic-optimization policy is native-controlled; the earlier
   mapper's fixed-intrinsic settings must not be claimed for this path.
8. Validate the native output before publication. Restore synthetic output names
   (`frame_<IMAGE_ID>.jpg`) from the input ID mapping only after checking camera IDs
   and unchanged observation coordinates/order. Validate reciprocal associations,
   positive camera depth and reprojection error. Retain tracks with at least three
   observations within 3 pixels; unsupported cameras remain explicitly missing.

The saved pose-graph stage is not independently applied to the VGGT depth seeds.
The standalone BA path uses the original VGGT pose/depth frame consistently; it
does not currently inject the pose graph as an optimization constraint. VGGT depth
is an initialization, not a persistent depth prior. This is not GTSfM's complete
hierarchical refinement or a dense-depth optimization method.

ALIKED is the only supported feature family in StereoForge. Native tensor models
and GPU/TensorRT-specific engines use a persistent content-keyed cache. Protobuf
field compatibility must be revisited when the pinned pyCuSFM binaries change.

## Artifacts, visualization and recovery

A full run publishes `data/intermediate/geometry_<timestamp>/` containing:

- `input_frames/`: decoded images and timestamps for video inputs.
- `geometry.npz`, `metadata.json`, `run.json`: merged VGGT arrays and provenance.
- `previews/`, `contact_sheet.jpg`, `point_cloud.ply`, `index.html`: dense inspection.
- `pycusfm/input/`, `config/`, stage logs, and `workspace/`: native refinement inputs
  and diagnostics.
- `pycusfm/initialized_sparse/` and `initialized_binary/`: depth-seeded BA models.
- `pycusfm/workspace/ba_raw/`: unmodified optimized export.
- `pycusfm/workspace/sparse/`: validated sparse model.
- `pycusfm/initialization.json`, `refinement.json`, `comparison.json`, `index.html`:
  initialization counts, enabled stages, coverage, alignment and sparse viewer.

The self-contained WebGL viewer supports points, camera frustums, trajectory,
progressive playback and RGB/depth/confidence previews. Playback reveals a saved
final reconstruction; it is not live reconstruction. The sparse viewer links back
to VGGT and labels the dense previews as VGGT references.

Sparse cameras are aligned to VGGT with a similarity transform for comparison.
This removes global gauge differences, not local drift. Missing cameras are not
filled with invented refined poses. `complete` requires full frame coverage plus
aligned nonempty geometry; it does not certify reconstruction accuracy. Partial,
empty, failed and interrupted results retain diagnostic explanations.

The full runner preserves the VGGT report if refinement fails. Recovery commands
can refine saved arrays, retry BA from initialized landmarks, or regenerate a
report from completed BA output. These create new result folders rather than
replacing earlier runs. A successful native exit code alone is insufficient;
reported failures and exported geometry are checked separately.

## Roadmap beyond geometry

The next work is to assess camera/landmark coverage, temporal consistency and
reprojection quality, then reconcile refined camera geometry with dense depth.
A refined sparse pose cannot safely be substituted into the original depth map
without accounting for their geometric relationship.

After that, implement the baseline controller, temporal filtering, StereoSpace
adapter and SBS encoder as real modules when each feature is developed.

For rectified pinhole stereo, the design relationship is:

```text
disparity_px = focal_length_px × baseline / depth
baseline_limit = disparity_limit_px × robust_near_depth / focal_length_px
```

Baseline and depth must share units. Choose a confidence-filtered near-depth
percentile and limit the preferred baseline by the disparity budget. Do not enforce
a minimum baseline that overrides that budget. Meter-valued preferences require
calibrated scale; unknown-scale operation needs an explicit relative-unit policy.
VGGT confidence must not be interpreted as a 0–1 probability.

Temporal control should limit abrupt baseline changes over the recording, with
explicit state reset at its start. StereoSpace will synthesize the second eye
from the source image and calibrated synthesis parameters; its exact API and
sequence consistency must be verified during implementation. Full SBS output
places left and right views horizontally, preserves timing, and should retain
source audio where feasible. None of these downstream behaviors is implemented yet.

## Maintenance rules

- Keep changes within the active scope; do not create empty future-stage packages.
- Preserve existing CLI commands while organizing implementation by responsibility.
- Keep native subprocess failure handling, geometric validation and recovery paths.
- Store datasets, checkpoints, credentials, caches and generated reports outside Git.
- Track this specification and keep it consistent with code and README commands.
- Do not modify third-party source to disguise integration mistakes.
- Add meaningful tests with implemented behavior when authorized; empty test files
  are not verification. The current user workflow reserves builds, tests and
  inference execution for the user.
