# StereoForge

StereoForge reconstructs one continuous video into **one camera trajectory and
colored sparse and refined dense point clouds**. The active scope is:

```text
Video → keyframes → measured feature tracks
      → overlapping VGGT windows → local cuNLS BA
      → Sim(3) initialization in one common map → joint cuNLS BA
      → shared-calibration global BA → dense refinement + fusion
      → camera trajectory + colored cloud + one viewer
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

After merging, one global BA jointly optimizes shared `fx, fy, cx, cy`, camera
poses and sparse landmarks. Focal lengths have a ten-pixel prior; principal points
have a conservative two-pixel prior per camera. The first camera remains fixed.
There is no intermediate focal-only acceptance gate. All cameras must have identical
processed dimensions; shared calibration assumes unchanged zoom/crop. A failed
solve or boundary check preserves the accepted sparse map and stops before dense
refinement. Status records `shared_calibration_complete` and the final intrinsics.
Optional Jacobian diagnostics cover all 13 reprojection tangent coordinates.
Final global BA still requires 80% held-out agreement within five pixels for every
boundary. Individual frames below 80% produce a lower-confidence warning with
before/after counts and median errors. For frames with at least five holdouts,
reject a drop exceeding 20 percentage points or a median error exceeding both
10 pixels and twice its pre-BA value; nonfinite medians also reject. Newly invalid
held-out projections reject independently of sample count. Missing landmarks
remain in the denominator and count as infinite error. Window-merge validation
retains its strict per-frame 80% check.

Dense refinement is a separate, conservative geometric pass using the final map.
Sparse observations calibrate each VGGT depth map to the map's arbitrary scale.
CUDA tensor operations check six nearby keyframes (offsets ±1, ±2, ±4): valid
confidence, positive depth, 5% depth agreement, two-pixel round-trip reprojection,
0.5-degree parallax and color agreement. Inverse-depth consensus updates are checked
again against the neighbors. At least two neighbors must support a pixel before it
enters the cloud. Unsupported pixels retain their scaled prior and an explicit
unsupported mask; this does not fill unseen surfaces or guarantee moving objects
are removed. Raw VGGT windows are preserved.

`dense/` contains calibrated priors, refined depth arrays, support masks and preview
images. C++ voxel fusion exports `dense_point_cloud.ply`; `point_cloud.ply` and the
main viewer show that same dense result, while `sparse_point_cloud.ply` preserves
BA landmarks. The viewer caps displayed points at 240,000; PLY exports all fused
voxels. `--dense-voxel-fraction` sets voxel width relative to median scene depth
(default 0.01). Fusion stops explicitly at five million voxels rather than silently
truncating output; increase this fraction to reduce memory. Status reports dense
coverage, unanchored frames, voxel size and point count. Failed dense refinement
preserves the sparse map. This stage uses PyTorch CUDA for dense projection/sampling,
C++ for fusion and custom CUDA/cuNLS for calibration BA; it has not been run by the
coding agent.

Window merging follows measured connections rather than timestamp order. All VGGT
windows are inferred first. C++ ranks seed candidates by tracks observed in at least
three window frames; subsequent candidates are ranked by shared cameras, then
existing-map track support. A rejected candidate is deferred without changing the
accepted map and is retried at most once per map revision. Windows without shared
cameras wait for a connection; tracks alone do not bypass the camera alignment.
When no candidate can advance the map, scheduling stops instead of retrying forever.

Accepted frame-to-window ownership selects depth priors for arbitrary merge order.
Python keeps a two-window tensor cache and loads only the accepted source windows
needed for the current overlap; it does not retain every dense window in RAM.
A final global cuNLS BA pass optimizes the assembled map and validates all stored
boundaries. Final status includes `merge_order`, `unresolved_windows` with reasons,
and `global_ba_complete`. Completion requires every selected frame and successful
final BA; incomplete coverage is published as partial. No disconnected component
is placed into the map with an invented transform. A failed bridge in a chain may
still prevent full coverage when there is no alternative connection.

Every window is initialized from shared camera poses. Relative camera orientations
provide a robust rotation estimate. Shared camera displacements supply scale when
at least three consistent pairs move more than 2% of scene depth in both maps.
During small translations or rotation, matching pixels in identical shared RGB
frames supply a median depth-ratio scale prior. Accepted source-window depths are loaded on demand; reprojection-consistent sparse landmarks calibrate each frame's
depth to its current BA gauge before ratios are compared. This is an approximate
scale prior, not corrected dense depth. Shared camera centers anchor translation.

Local windows may contain multiple connected camera/track groups. Each input
group receives its own local BA solve, and groups are recomputed after outlier
filtering. Each surviving group is independently aligned to the unchanged accepted
map using its shared cameras. A single shared camera can anchor rotation and
translation when shared-frame depth supplies scale. A group with no shared camera
or usable scale evidence is rejected with its frame range and shared-camera count;
it is not silently dropped. The seed must be connected, and combined-map
connectivity, per-camera support and post-BA overlap checks remain mandatory.

Overlap validation requires at least 40 held-out observations overall. A shared
camera with fewer than five holdouts is marked as insufficiently validated in a
saved warning, rather than stopping BA. Its observations still count toward the
80% overall agreement requirement. Per-camera 80% checks apply when at least five
holdouts exist, including at later merges; camera preservation and connectivity
checks remain mandatory.

There is no PnP recovery branch or mandatory 60-landmark triangulation-angle gate.
Measured feature tracks still connect the windows for robust joint BA. Rotation
disagreement is reported as an alignment warning; absent scale evidence, invalid
transforms, disconnected maps and failed post-BA overlap checks still reject a
candidate. Warnings are retained in `status.json`. The established map is preserved
on rejection. Depth/RGB caching is bounded; raw VGGT tensors remain on disk.

`--device cuda` uses visible GPUs for VGGT windows and the first visible GPU for
BA. `--device cuda:0` selects one GPU. Single-GPU systems use the same pipeline.
VGGT workers exit before BA begins. `--lm-iterations 300` is the default joint
solve budget; `--neighbors 4` controls temporal feature matching.

Open `data/intermediate/reconstruction_TIMESTAMP/index.html`. The same directory
contains:

- `trajectory.json`: camera-to-world poses, intrinsics and video timestamps.
- `point_cloud.ply`: the fused dense cloud after successful dense refinement.
- `sparse_point_cloud.ply`: optimized sparse landmarks retained for comparison.
- `dense/`: refined depth arrays, support masks and previews.
- `status.json`: completion status, registered/missing keyframes and failing stage.
- `selection.json`: selected keyframes mapped to original video frames/timestamps.
- `vggt/`: initial per-window depth, calibration and camera estimates.
- `map_TIMESTAMP/`: attempt settings only; no automatic COLMAP export.

Distances are in reconstruction units, not established meters. Sparse BA does
not itself refine VGGT's dense depth; the separate dense pass does that on supported
pixels. Stereo synthesis
and video encoding remain outside this scope.

Rejected windows are deferred while connected alternatives are tried. If coverage
remains incomplete, the viewer shows the **partial map** and lists missing frames.
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
are included in the final status, including when a window fails. Camera-based
Sim(3) is an initialization for BA; it does not correct internal window distortion.
The implementation has not been compiled or run by the coding agent.

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

The script builds the CUDA → base → geometry → stereo → cuNLS → Rust/Rerun image chain with
BuildKit and starts an interactive container. If the named container is already
running, it attaches without rebuilding. Exit/stop it before rebuilding changed
Dockerfiles or native code. Python source and configuration changes are visible
through their explicit workspace mounts.

The cuNLS image builds the unchanged `third_party/cuNLS` checkout against CUDA 13.2.
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

The active pipeline uses `windowed.py`, `native_map.py` and the C++ MapBuilder. The 26 legacy experiment and pyCuSFM source files have been removed. See
[StereoForge Architecture Specification.md](StereoForge%20Architecture%20Specification.md)
for conventions and implementation details. The consolidated path is implemented
but has not been run or compiled by the coding agent.

## How windows share a map

RaCo–ALIKED descriptors and LightGlue+ establish measured image matches, filtered
by RANSAC and assembled into tracks. VGGT depths initialize a 3D landmark for
each supported track in each window. Overlap observations identify the same
tracks across windows; no descriptor matching of point-cloud coordinates or ICP
is needed. Sim(3) uses shared camera orientations and centers, with shared-frame
depth ratios as a scale prior during low translation. Joint cuNLS BA then refines the
combined map using measured 2D observations. Similarity alignment initializes BA;
it does not by itself correct internal distortion.

The final `docker/Dockerfile.rerun` layer installs Rust stable (Cargo, rustfmt,
Clippy) and the official Rerun 0.38.1 viewer binary, checked against its release
SHA-256 digest. No Rerun source is modified. Rebuild after stopping the existing
container, using `bash scripts/build_and_start.sh`. Inside the new container,
`rerun` opens the viewer using the existing display mounts. This installs the
environment only: reconstruction logging is not connected yet. Our future Rust
visualization package will declare the matching `rerun` crate in `Cargo.toml`;
no Python or C++ Rerun SDK is installed by this layer. Cargo's registry cache and
Rust toolchain currently live in the container and are not persisted on the host.
