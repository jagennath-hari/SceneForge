# StereoForge

StereoForge reconstructs one continuous video into **one camera trajectory and
colored sparse point cloud**. The active scope is:

```text
Video → keyframes → measured feature tracks
      → overlapping VGGT windows → local cuNLS BA
      → Sim(3) initialization in one common map → joint cuNLS BA
      → camera trajectory + colored sparse cloud + one viewer
```

## Run end to end

Inside Docker:

```bash
python -m stereoforge.reconstruction \
  --video data/input/barn.mp4 \
  --window-size 32 --overlap 8
```

Window size and overlap count **selected keyframes**, not original video frames.
The final window may be shorter and retains every remaining keyframe. Overlap
must be at least six and smaller than the window size. Each window is refined
locally, then added to the current common map using shared cameras and measured
track identities. Joint BA uses a three-pixel Huber loss; local BA uses GNC-TLS.
The first window establishes the coordinate system. Shared keyframes have one
camera pose, and shared tracks are fused into one landmark. Reconciliation keeps
established landmark positions and observations, admitting new observations only
within four pixels of their predicted position. A rejected extension cannot delete
its established track. Joint BA then refines the map, subject to current and older
overlap validation; its normal outlier filtering still applies.

When shared depth support is insufficient for Sim(3), the native builder attempts
PnP/RANSAC against existing global map landmarks. This fallback is not triggered
by a fitted transform failing its geometry checks. New cameras are registered in
frame order, retaining established landmark positions until joint BA. Measured
tracks supply the correspondences; there is no new descriptor-matching pass.

Recovery uses C++ OpenCV PnP and Eigen triangulation on CPU, followed by CUDA/cuNLS
joint BA. Each new camera needs at least 30 training and 10 held-out correspondences,
at least 20 training inliers covering 50% of training candidates, and at least 80%
held-out agreement within five pixels. Concentrated features do not block PnP;
there is no image-grid coverage requirement. Held-out pixels stay excluded from BA and are rechecked at later merges.
New landmarks require three registered views, positive depth, at most four-pixel
reprojection error and at least one degree of ray separation. Existing tracks are
not replaced by incoming VGGT depths. Recovery commits only after all incoming
cameras and the existing overlap checks pass; otherwise the accepted map survives.
This is a fallback for insufficient alignment support after local BA succeeds,
not a recovery for failed VGGT/local BA or complete feature-tracking loss.

`--device cuda` uses visible GPUs for VGGT windows and the first visible GPU for
BA. `--device cuda:0` selects one GPU. Single-GPU systems use the same pipeline.
VGGT workers exit before BA begins. `--lm-iterations 300` is the default joint
solve budget; `--neighbors 4` controls temporal feature matching.

Open `data/intermediate/reconstruction_TIMESTAMP/index.html`. The same directory
contains:

- `trajectory.json`: camera-to-world poses, intrinsics and video timestamps.
- `point_cloud.ply`: all optimized colored sparse landmarks.
- `status.json`: completion status, registered/missing keyframes and failing stage.
- `selection.json`: selected keyframes mapped to original video frames/timestamps.
- `vggt/`: initial per-window depth, calibration and camera estimates.
- `map_TIMESTAMP/`: attempt settings only; no automatic COLMAP export.

Distances are in reconstruction units, not established meters. Sparse BA does
not refine VGGT's dense depth. Full-video dense reconstruction, stereo synthesis
and video encoding remain outside this scope.

A rejected window stops map growth. If a validated prefix exists, the viewer
shows that **partial map** and lists missing frames; it is never marked complete.
The previous accepted map remains intact when a candidate merge fails. Validation
checks stay enabled, including checks on previously merged boundaries.

## Optional diagnostics and retry

The common map stays inside one C++ object, called through pybind11. Eigen handles
landmark initialization, Sim(3), merging and validation on the CPU; custom CUDA and
cuNLS handle local/joint BA. Options, statistics and results are typed C++ structs;
the native map/solver interface contains no JSON objects. Typed buffers go directly into BA without per-window
JSON requests, subprocesses, COLMAP round trips or intermediate viewers. Track
identities remain stable; earlier-boundary validation uses indexed lookups.
Python loads VGGT tensors and exports the final map once. Input caches and final
viewer/trajectory/status metadata remain; they are not solver interchange files.

Add `--diagnostics` for CUDA Jacobian checks; `--debug` adds a traceback.
The progress bar names the current native stage. Aggregate native stage timings
are included in the final status, including when a window fails. No speedup has
yet been measured. Eigen's Sim(3) refinement optimizes the same seven-parameter
objective, but its LM implementation and deterministic RANSAC samples differ from
SciPy/NumPy; results are not guaranteed to be bitwise identical. Geometry and
held-out validation thresholds remain unchanged.

To reuse unchanged inputs, feature tracks and VGGT windows from a run of this
new pipeline, recomputing BA in a new map attempt:

```bash
python -m stereoforge.reconstruction \
  --resume data/intermediate/reconstruction_TIMESTAMP
```

`--lm-iterations` and diagnostics may change on resume. Video, window/matching
settings, checkpoint and implementation must match the saved request. Older
experimental runs cannot be resumed through this command; start with `--video`
to reuse the video/keyframe cache. Old results are not deleted.

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
`find_package(cunls CONFIG REQUIRED)` exposes `cunls::cunls` for our native solver.
The image also installs Eigen3 and builds `_stereoforge_map` into the Python venv.
Rebuild the image after native changes; old images cannot run the new map builder.

The active reconstruction command uses our custom cuNLS C++/CUDA backend.
pyCuSFM is no longer installed. Older experimental commands that require it are
not part of the current workflow.

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

## Source organization

- `stereoforge/reconstruction/__main__.py`: the supported end-to-end command.
- `stereoforge/reconstruction/windowed.py`: window scheduling and common-map orchestration.
- `stereoforge/optimization/include/` and `src/`: custom cuNLS C++/CUDA factors and solver.
- `stereoforge/video/`: Python adapters, plus C++/CUDA sources under `include/` and `src/`.
- `stereoforge/geometry/`: VGGT inference and geometry types.
- `stereoforge/refinement/`: sparse models and cuNLS process adapter.

The active pipeline uses normal `cluster.py` and `merging.py` modules, not demo
entry points. The 26 legacy experiment and pyCuSFM source files have been removed. See
[StereoForge Architecture Specification.md](StereoForge%20Architecture%20Specification.md)
for conventions and implementation details. The consolidated path is implemented
but has not been run or compiled by the coding agent.

## How windows share a map

RaCo–ALIKED descriptors and LightGlue+ establish measured image matches, filtered
by RANSAC and assembled into tracks. VGGT depths initialize a 3D landmark for
each supported track in each window. Overlap observations identify the same
tracks across windows; no descriptor matching of point-cloud coordinates or ICP
is needed. Sim(3) fits corresponding 3D landmarks and shared camera constraints
to initialize rotation, translation and scale. Joint cuNLS BA then refines the
combined map using measured 2D observations. Similarity alignment initializes BA;
it does not by itself correct internal distortion.
