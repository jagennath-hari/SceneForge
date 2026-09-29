# SceneForge Architecture Specification

## Scope

Reconstruct one continuous video into a common 3D map: one optimized camera per
selected keyframe, colored sparse landmarks, refined depth maps and a fused dense
point cloud. Rerun provides live visualization and recording.

The active pipeline is:

```text
Video
  → streaming decode and keyframe selection
  → joint feature-image/global-descriptor preparation
  → unified temporal and loop pair verification
  → measured feature tracks and bounded connection repair
  → overlapping VGGT-Ω windows
  → local bundle adjustment and Sim(3) registration into a common map
  → shared-calibration global bundle adjustment
  → dense depth calibration, refinement and voxel fusion
```

This specification describes the current implementation. Loop closure adds
verified long-range landmark observations to BA. Stereo pair generation and video
encoding are outside scope. The pipeline uses Sim(3), not projective SL(4) deformation.

## Code ownership

| Directory | Responsibility |
| --- | --- |
| `sceneforge/video` | FFmpeg/NVDEC decoding, RaCo–ALIKED/LightGlue+ TensorRT inference, CUDA preprocessing/RANSAC, keyframe and model caches |
| `sceneforge/geometry` | VGGT-Ω adapter, checkpoint resolution, tensor contracts and per-window storage |
| `sceneforge/reconstruction` | Python orchestration, global retrieval, temporal/loop tracks, connection repair, window scheduling, output snapshots and dense refinement |
| `sceneforge/optimization` | C++/Eigen map operations, custom CUDA factors, cuNLS BA and pybind11 interface |
| `sceneforge/visualization` | Rust Rerun SDK adapter behind a C ABI |
| `sceneforge/utils` | Progress reporting, validation, camera conversion and artifact export |

Native public headers live under each component's `include/sceneforge/` tree;
implementations live under `src/`. Custom CUDA kernel declarations/implementations
use `.cuh`/`.cu`; host C++ uses `.hpp`/`.cpp`. C++ uses explicit types and `this->`
for its own class members. Upstream submodules are consumed without source edits.

The common map and optimization requests stay in memory as typed native objects.
Python loads VGGT-Ω tensors and exports snapshots; it does not exchange solver state
through COLMAP files or a separate file-based BA executable.

## 1. Video and keyframes

`VideoFrameSampler` invokes the native streaming selector. An ordered decoder
feeds a bounded queue of image buffers to feature workers. Feature extraction can
use multiple peer-accessible GPUs, while matching and acceptance decisions remain
in timestamp order against the last accepted keyframe. One GPU follows the same
path. The decoder-to-feature handoff currently uses host buffers.

RaCo–ALIKED and LightGlue+ run from cached TensorRT engines. CUDA geometric
verification estimates fundamental/homography support for keyframe selection.
The active settings are in `configs/keyframes_raco.json`.

Only accepted images are PNG-encoded. Candidate metadata retains source-frame IDs
and timestamps, allowing connection repair to seek specific intermediate frames
from the original video. The source video must remain available and unchanged.

Caches use `.keyframes/` beside the container's source path (the host launcher
mounts `.cache/sceneforge/video-keyframes/` there) and include source metadata,
settings, model identity and the selector executable in their identity. Locked,
atomic publication prevents reuse of incomplete results. Standalone frame
extraction and the native selector's debug view remain available for inspection.

## 2. Measured image evidence

Feature images use VGGT-Ω's balanced preprocessing at resolution 512. Pixel
coordinates, intrinsics and feature tracks must refer to that processed grid.
All input frames must have uniform processed dimensions.

Feature preparation reads each source RGB once, using a bounded two-worker CPU
queue. The pinned VGGT-Ω crop/shape helpers preserve the existing geometry grid.
A separate retrieval branch applies SelaVPR++'s RGB normalization and bilinear
322×322 hard resize. Geometry pixels and intrinsics never use this retrieval grid.

The local SelaVPR++ Hub implementation supplies DINOv2-base + GeM with hashing and
reranking disabled. Eight-image GPU batches produce normalized 2,048-dimensional
float descriptors. Cache identity includes image contents, checkpoint hash, model
source and preprocessing/runtime settings. The model is unloaded before matching;
a repair round infers descriptors only for missing images.

Cosine retrieval runs in bounded GPU blocks. For each frame, at most three
candidates are chosen outside both the 32-keyframe exclusion neighborhood and a
ten-second time separation. Candidate regions are spaced by five seconds. Ranking
uses frame IDs to resolve equal scores. This bounds candidate count but does not
promise bitwise reproducibility across GPU/software versions.

The frontend combines deduplicated retrieval candidates with each frame's next
four temporal neighbors. One verification queue uses native RaCo–ALIKED/LightGlue+
and RANSAC. Native feature buffers use bounded LRU reuse within each batch. Cached
pair results are checked against the current pair plan, and repair rounds remap
measurements by immutable source identity.

Loop pairs must satisfy geometric support and spatial coverage, then be
corroborated by another independently verified nearby pair. The current defaults
are 30 inliers, 25% inlier ratio, four occupied cells of a 4×4 grid in each image,
and a neighboring pair within two frames at both endpoints. Compact cross-batch
decisions avoid rewriting the full match files. Candidate, geometry-verified and
corroborated counts are recorded separately.

`VerifiedGraph` joins consistent temporal and accepted loop edges into measured
tracks, with at most one feature observation per frame. Conflicting edges are
rejected without discarding already consistent observations. Tracks with at least
three views supply reconstruction measurements. Loop provenance is retained by
track ID and endpoint frame pair.

An independent temporal-only track forest is built from the same parsed matches.
Connection repair uses its crossing-track counts, so distant loops cannot hide
weak local boundaries. Repair may insert real intermediate frames, reuse old
descriptors/measurements and match new pairs. Remaining weak cuts and chronological
frame mappings are reported. Image-graph connectivity alone does not guarantee
observable 3D geometry.

Verified loops constrain optimization only when their endpoints survive as
observations of the same native landmark. Final reporting counts those retained
landmarks, loop pairs and observations separately from frontend track counts.
This implementation does not add a separate pose graph or change the map's
alignment/BA acceptance policy. `--no-loop-closure` disables retrieval for comparison.

## 3. VGGT-Ω windows

Default windows contain 64 selected keyframes with 32 shared keyframes. The final
window may be shorter. These dimensions are configurable through the CLI.

Independent workers distribute windows across visible GPUs. Each worker loads
VGGT-Ω once and publishes CPU tensors atomically. Workers finish before native
map optimization begins. GPU memory is not pooled between devices.

Each saved window contains processed RGB, depth, confidence, intrinsics and
camera poses. These are initialization estimates in a window-local coordinate
system; they are not assumed to agree exactly with another window.

The gated checkpoint is resolved from the persistent Hugging Face cache, with an
automatic authorized download when missing. Explicit checkpoints remain supported.

## 4. Building the common map

The native builder initializes sparse landmarks from VGGT-Ω depth and measured
feature tracks, then performs local cuNLS BA. Surviving connected groups determine
which geometry can be proposed to the accepted map.

Windows are scheduled by measured support and shared cameras, rather than forcing
chronological insertion. The first accepted connected window establishes the
coordinate system. Later windows use a similarity transform:

```text
X_map = s R X_window + t,    s > 0, R in SO(3).
```

Shared-camera orientations constrain rotation. Camera displacements constrain
scale when translation is sufficient; shared-frame depth ratios provide a scale
prior when it is not. Shared camera centers constrain translation. Sim(3)
reconciles coordinate systems but cannot repair arbitrary internal distortion.

Combining maps keeps one camera per keyframe and associates landmarks through
measured track identity. Established landmarks and observations are retained;
new observations are proposals whose consistency is checked before admission.
Joint BA refines the candidate map. Boundary checks decide whether to commit it,
preserving the accepted map when a candidate is rejected.

The existing recovery mechanisms are bounded: disconnected local components can
use measured 2D-to-3D registration; supported subsets and smaller overlapping
predictions can be proposed when a whole window is unusable. Missing cameras stay
pending and are reported if no connected proposal can add them. Recovery does not
invent a transform for an unconnected group.

Eigen handles CPU geometry, map combination and validation. Custom CUDA factors
and cuNLS handle BA. Local optimization avoids repeatedly solving the entire map;
periodic global refinement and the final global pass address accumulated changes.

## 5. Global optimization

After map assembly, final global BA jointly optimizes camera poses, sparse point
positions and shared `fx`, `fy`, `cx`, `cy`. Shared calibration assumes a fixed
camera/lens and unchanged effective crop. The coordinate-system anchor is retained.

Measured reprojection residuals are robustified, with calibration and pose priors
as defined by the native solver. Local BA uses GNC-TLS; joint BA uses Huber loss.
The iteration budget bounds runtime. A finite, non-increasing-cost joint result
may proceed when it reaches that budget; reaching it does not establish convergence.

Window acceptance still validates geometry and overlap. Final global BA reports
held-out reprojection regressions as warnings rather than making every frame's
percentage a veto. Solver failure and invalid geometry remain errors. No threshold
or optimization policy is changed by source cleanup.

## 6. Dense refinement

Sparse BA optimizes landmarks, not every depth pixel. A separate pass calibrates
each owned VGGT-Ω depth map to the final sparse map's scale. Raw VGGT-Ω windows remain
unchanged.

PyTorch CUDA operations compare neighboring views using positive depth,
confidence, relative depth agreement, reprojection consistency, parallax and color.
Supported pixels receive inverse-depth consensus updates. Unsupported pixels
retain their calibrated priors with explicit support masks and do not enter the
fused cloud.

Native voxel fusion combines supported colored points. Voxel width is relative
to median scene depth (`--dense-voxel-fraction`, default 0.01). Dense output is
partial surface evidence: it does not fill unseen surfaces or guarantee removal
of every moving object or bad prediction.

## Coordinate and data contracts

- Camera optical axes: **X right, Y down, Z forward**.
- `FrameGeometry.camera_to_world`: rigid `[4,4]` transform.
- Native BA uses world-to-camera poses; conversion occurs at the adapter boundary.
- Intrinsics: `[3,3]` pinhole matrix in processed-image pixels, including `fx, fy, cx, cy`.
- Depth: `[H,W]` optical Z depth, not Euclidean ray length.
- Processed RGB: `[N,3,H,W]`; confidence and depth share the same image grid.
- `SparseCamera`, `SparsePoint`, `SparseModel` in `reconstruction/models.py` are
  export snapshots. The mutable optimization state remains native.
- Source video indices, selected keyframe indices and window-local indices are
  distinct. Manifests retain their mapping and timestamps.
- Scale is arbitrary unless externally established. Outputs use
  `reconstruction_units`; they must not be interpreted as meters.

## Visualization

C++ invokes the SceneForge Rust adapter; the adapter uses the unmodified Rerun
Rust SDK to stream to the standard viewer and save a `.rrd` recording.

Early cameras arranged on a sphere are schematic placeholders. Feature links and
track activity visualize processing, not recovered poses. VGGT-Ω windows are shown
as separate local predictions until accepted into the common map. Accepted sparse
geometry, calibration updates and dense fusion then update the same viewer.

Display sampling and camera motion are visualization choices. PLY export preserves
the full exported cloud independently of the viewer's point limit.

## Storage and execution

Run from the host with:

```bash
bash scripts/build_and_start.sh data/input/barn.mp4
```

The launcher requires host `ffprobe`, validates a short prefix of the video before
building Docker, mounts only the chosen input file read-only at `/input/video`,
and executes `python -m sceneforge.reconstruction --video /input/video`. Relative
paths resolve against the caller's directory; files outside the repository are
supported. The writable keyframe cache is mounted separately. The container exits
with the reconstruction process rather than opening a shell.

`data/input/` holds source videos. New runs live in
`data/output/reconstruction_TIMESTAMP/`, with final artifacts at the run root:
`status.json`, `trajectory.json`, `point_cloud.ply`, and, after dense success,
`sparse_point_cloud.ply` and `dense_point_cloud.ply`.

Run-local input manifests, processed images, matches, tracks, repair records,
VGGT-Ω tensors and `dense/` arrays are retained for reuse and inspection.
`map_TIMESTAMP/` identifies each map-building attempt and contains its settings
and `pipeline.rrd`. There is no separate `data/intermediate` directory for new runs.

`--resume PATH` reuses inputs only when the stored implementation/input identity
matches, then builds a fresh map attempt. After code changes, start a new video
run and reuse compatible keyframe/model caches. Partial results explicitly list
missing frames; full coverage does not by itself establish reconstruction accuracy.

Docker uses BuildKit and separate base, geometry, cuNLS and Rerun images. The start
script supplies the runtime user, working directory and reconstruction command.
Python runs in one uv-managed environment. First-party native extensions and the
Rust adapter are built into the image; source changes to those components require
a rebuild. Datasets, generated runs and credentials are excluded from image builds
and Git. Upstream dependencies remain pinned by their submodule commits.
