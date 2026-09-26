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
python -m stereoforge.reconstruction --video data/input/barn.mp4
```

Defaults are `--window-size 32 --overlap 16 --lm-iterations 500`, with Rerun enabled.
Window size and overlap count **selected keyframes**, not original video frames.

`configs/keyframes_raco.json` is the active configuration file for RaCo–ALIKED +
LightGlue keyframe selection and engine preparation. Reconstruction window/BA
settings use the CLI defaults above and can be overridden with flags. The obsolete
`default.yaml` demo/pyCuSFM configuration and its loader have been removed.
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
VGGT workers exit before BA begins. `--lm-iterations 500` is the default joint
solve budget; `--neighbors 4` controls temporal feature matching.

Rerun opens automatically and records to
`data/intermediate/reconstruction_TIMESTAMP/map_TIMESTAMP/pipeline.rrd`.
Use `rerun /path/to/pipeline.rrd` for replay, or `--no-rerun` for headless processing.
New runs do not generate HTML. The output directory contains:

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

The final `docker/Dockerfile.rerun` layer installs Rust tooling and the official
Rerun 0.38.1 viewer (release SHA-256 checked), then builds StereoForge's Rust
visualization adapter using the matching official Rust SDK. Rerun itself is
unmodified; no Rerun C++ or Python SDK is used. Stop the old container and rebuild
with `bash scripts/build_and_start.sh` before using it.

Launch the live viewer and record reconstruction stages with:

```bash
python -m stereoforge.reconstruction --video data/input/barn.mp4 \
  --window-size 32 --overlap 16 --lm-iterations 500 --rerun
```

`--rerun` is enabled by default and launches the official viewer (or connects to one already
running), streams through gRPC and simultaneously saves the same recording to
`map_<attempt>/pipeline.rrd`. The GUI uses the existing Docker display mounts.
One 3D view shows the whole process, alongside a stage log. Accepted keyframes
appear incrementally as image planes on a deterministic gray grid; rejected
candidate frames are not rendered. This grid uses arbitrary display positions and
a fixed display FOV, not estimated poses/intrinsics. Cache hits populate the grid
from the selected keyframes. Image preparation updates the thumbnails to the
processed image grid. Matching and track building leave these 3D image planes in
place and update stage text; there are no 2D match panels or feature overlays.

Completed VGGT windows appear as separate normalized groups in the same 3D view.
Their offsets and uniform display scale only arrange the presentation; relative
positions between groups are not reconstructed yet. Sim(3)-aligned windows then
appear provisionally at their estimated location in the common map. Successful
merges remove the corresponding staged group and update the accepted map. Final
BA leaves only the accepted map. These are discrete stage updates, not fabricated
intermediate camera motion. No separate viewer command is needed for live use.
Select `reconstruction_step` and the end-of-timeline control to follow new updates;
scrubbing backward lets you inspect earlier stages. Stage flushes are asynchronous.

For replay after completion, use `rerun /path/to/pipeline.rrd`. Rerun 0.38 does not
follow growing recording files; live visualization uses the gRPC connection, not
file tailing. An already-running reconstruction using the old file-only adapter
must finish or be restarted with the rebuilt image to use live streaming.

`world/map` contains accepted colored points and trajectory. Persistent camera
entities at `world/cameras/<keyframe>` start on an inward-facing schematic sphere,
move into normalized VGGT groups, then move into the accepted common map.
Overlapping windows reuse those entities; accepted poses are not overwritten by
later provisional windows. Completed window point previews are retired.
Gray cameras have arbitrary display FOV until VGGT supplies calibration.

The viewer uses one full-width 3D scene, without a pipeline-status pane.
An unlabeled origin marker and red X, green Y, blue Z axes appear immediately and
remain visible throughout processing. This is a normalized display reference,
not a surveyed origin or metric scale. Inspection
panels and the timeline start collapsed. Logs remain recorded under `pipeline`.
During image preparation, the active camera is highlighted amber and becomes
green when its processed thumbnail is ready. Matching displays up to 16 requested
pair links in amber, then a sample of geometrically verified links in green after
that batch completes (including reused batches). Track assembly shows up to four measured tracks with eight observations each.
Colored markers lie at their measured pixel locations on the processed image
planes, with same-colored links joining each track. Pixel coordinates are scaled
with the thumbnail dimensions and projected using the same schematic camera pose
and display FOV as the image plane. These links are not triangulated 3D landmarks.
The status label reports processed edges or collected tracks/components, without
inventing camera-completion counts. One in-scene label names
the current operation, including model preparation; it does not invent progress
when the underlying operation exposes none. Activity replaces earlier highlights
and is cleared when a VGGT/map snapshot arrives.

Camera icons stay the same display size. Each map update is centered and uniformly
scaled into a bounded display volume; this changes display coordinates only, not
saved camera poses, depth, Sim(3), BA or PLY geometry. Extreme point outliers are
excluded from the preview only. Independent VGGT groups sit around the same origin.
Camera positions update at pipeline snapshots, without interpolated animation.

Camera-to-world rotations and RDF optical axes (+X right, +Y down, +Z forward)
are preserved. Intrinsics are scaled to image thumbnails, including the calibrated
principal point. World RDF does not imply gravity alignment. Previews cap points
at 50,000 per snapshot and images at 256 pixels per side.

Window merging requires 75% per-camera held-out agreement and 80% across the
boundary; the 5-pixel residual cutoff, positive-depth checks and final global BA
regression checks remain in place.

Logging can be disabled with `--no-rerun`, uses the official Rust SDK through a C ABI, and does not
change reconstruction decisions. No Rerun source is modified. Visualization errors
warn and disable logging. Dense calibration/refinement highlight the active optimized camera. Every eight
frames, a bounded preview of validated dense samples updates in the same scene;
after fusion it is replaced by the final voxel-fused preview (up to 240,000 points).
Dense and sparse points share the final map display transform. Full-resolution
clouds remain in PLY exports. Global BA displays a solve status, then updates all
calibrated frustums on acceptance; solver-iteration states are not streamed.

The Rerun viewport initially follows a separate presentation camera via
`EyeControls3D`. It frames the schematic sphere from an oblique angle, then fits
accepted global-BA and dense snapshots to their visible bounds. It never changes
reconstruction cameras. Updates follow geometry snapshots rather than a timed
orbit; manual navigation can detach from tracking. Track `world/presentation_eye`
again to resume following. The presentation camera is excluded from scene content
so it does not inflate bounds or appear as another reconstructed camera.

During dense calibration and refinement, the presentation eye follows behind and
slightly above the active optimized camera. Position and direction are smoothed,
with a fixed lens/offset and stable up reference. Dense previews do not trigger
bounding-box zoom changes. Completion returns to the whole-scene overview.
These remain discrete processing updates, not interpolated animation frames.
