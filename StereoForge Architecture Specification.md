# StereoForge Architecture Specification

## Product intent and current scope

StereoForge will convert a monocular video into stereoscopic side-by-side video.
Its central idea is to use scene geometry to choose a virtual stereo baseline
that adapts over time, then synthesize the second eye with StereoSpace.

The current deliverable is **inspectable geometry, sparse BA and supported dense refinement for one
continuous, uncut recording**. Drone footage and walkthroughs are the intended
inputs. Edited movies, shot detection, live visualization, baseline control,
StereoSpace inference, and video encoding are outside the implemented scope.
Their intended behavior is described in the roadmap, not represented by empty
Python modules or placeholder tests.

The supported command is `python -m stereoforge.reconstruction --video
PATH --window-size 32 --overlap 8`. The current pipeline is:

```text
Continuous video → ordered keyframes → measured global feature tracks
    → overlapping VGGT-Ω windows → local cuNLS BA
    → initialize each window in the first window's gauge with Sim(3)
    → merge shared cameras/tracks → joint robust cuNLS BA
    → shared-calibration global BA → dense refinement + fusion
    → one camera trajectory, colored cloud and viewer
```

The map grows in graph-ranked window order. Each global keyframe has one camera, each fused
track one sparse landmark. Local GNC-TLS filters initialization outliers; joint
BA uses a three-pixel radial Huber loss in the custom CUDA pixel factor. Shared
camera support and current/earlier withheld-observation checks remain
mandatory before a candidate replaces the accepted map. Priors remain those of
the initialized solve; this is not an unanchored global optimizer or an exact
GTSfM reproduction. Non-overlap child landmarks are preserved during reconciliation.

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

`--window-size` and `--overlap` count keyframes. All keyframes are covered, with a
possibly shorter final window. VGGT inference is scheduled across visible GPUs
(or one selected device) and workers exit before sequential cuNLS BA begins.
Single-GPU operation uses the same code. A persistent C++ MapBuilder owns cameras,
global landmark IDs, observations and validation boundaries. Eigen implements depth
initialization, camera-based Sim(3), track extension and validation on CPU.
Local and joint BA use typed in-memory buffers with the existing CUDA/cuNLS factors.
Python loads VGGT tensors, verifies processed image grids, invokes the native builder
through pybind11 and publishes the final map. No per-window solver JSON files,
subprocesses or COLMAP exports are used, including at final publication. BA options,
statistics and results use typed C++ structs, with no JSON dependency in the native
map/solver library. The legacy standalone JSON adapter is separate from this path.
The previous landmark-RANSAC/seven-parameter alignment and PnP recovery have been
removed from the native map builder. Shared camera orientation averaging and
robust center/depth scale statistics now initialize Sim(3). The production path
does not instantiate or call pyCuSFM. This change has not been compiled or run by
the coding agent.

The default output has one viewer, `trajectory.json`, refined-depth artifacts and
a voxel-fused `point_cloud.ply`; the sparse BA cloud is retained separately. Units are arbitrary reconstruction units. Raw VGGT depth remains an initialization artifact. Dense refinement is a
separate post-BA pass; only multi-view-supported pixels enter the fused cloud. `--diagnostics` enables CUDA Jacobian checks. One progress bar shows
the native stage; final status includes aggregate stage timings and a failure reason.
Input caches and final output metadata remain, without per-window diagnostic dumps.

`--resume` reuses immutable prepared inputs and VGGT windows of this pipeline,
then recomputes the map in a fresh attempt. It does not reuse previously optimized
maps under potentially changed priors. Input/settings/checkpoint/implementation
changes require a fresh run. Failed merges preserve and publish a labelled partial
map if a validated prefix exists. Missing cameras prevent complete status.

The earlier experimental paths below document development history. The active
entry point is `reconstruction/__main__.py`, using `native_map.py` to invoke the
C++ implementation under `optimization/`. The earlier Python cluster/merging helpers
are not on the production path.
The 26 approved legacy source files have been removed; earlier experimental
sections below are historical records, not supported commands. Dense/stereo rendering
and full-sequence runtime validation remain pending. No compilation, inference or
tests were run by the coding agent while consolidating this path.

## Repository structure

```text
StereoForge/
├── README.md
├── StereoForge Architecture Specification.md
├── configs/keyframes_raco.json      # Active feature/keyframe settings
├── docker/                         # BuildKit image chain, including cuNLS
├── scripts/build_and_start.sh
├── stereoforge/
│   ├── reconstruction/
│   │   ├── __main__.py              # The end-to-end command
│   │   ├── windowed.py              # Ordered windows and common-map growth
│   │   ├── frontend.py              # Image preparation, matching and final viewer
│   │   ├── view_graph.py            # Verified feature tracks and frame identities
│   │   ├── inference.py             # VGGT window scheduling
│   │   ├── cluster.py               # Depth initialization and local cuNLS BA
│   │   ├── merging.py               # Common-map fusion and joint cuNLS BA
│   │   ├── registration.py          # Shared-landmark correspondence and validation
│   │   ├── similarity.py            # Scale-aware 3D correspondence fit
│   │   ├── camera_alignment.py      # Shared-camera constraints on Sim(3)
│   │   ├── triangulation.py
│   │   └── merge_validation.py      # Current and earlier boundary checks
│   ├── geometry/                   # VGGT adapter, types, input/config helpers
│   │   ├── checkpoint.py            # Authorized checkpoint/cache resolution
│   │   └── storage.py               # Window tensor serialization
│   ├── refinement/
│   │   ├── cunls.py                 # Custom native BA process adapter
│   │   └── sparse_model.py          # Cameras, points, observations and COLMAP IO
│   ├── optimization/               # C++/CUDA solver; include/ and src/
│   ├── video/                      # Python adapters plus C++/CUDA include/ and src/
│   └── utils/                      # Reports, progress, validation and camera math
├── third_party/                    # Pinned upstream dependencies; unchanged
├── data/                           # Inputs and saved results; ignored
├── .cache/                         # Models/engines; ignored
└── .secrets/                       # Runtime credentials; ignored
```

Package `__init__.py` files identify real packages. Empty future-stage modules,
empty tests, and inactive scripts are not part of the source tree. Python is run
from the mounted workspace; dependency installation belongs to Dockerfiles and
`docker/constraints.txt`, rather than an empty packaging manifest.

## Environment boundary

The image chain is CUDA → base → geometry → stereo → cuNLS. Each Dockerfile has
an explicit named stage and accepts `BASE_FROM`. The current base is
`nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. PyTorch uses the CUDA 13.2 wheel index;
TensorRT retains its working version for native keyframe inference. cuNLS is built
from the pinned, unchanged submodule using this image's CUDA toolkit. Its shared
C++ library, headers and CMake config install under `/opt/cunls`; cuDSS libraries
from the matching upstream dependency archive install under `/opt/cudss/lib`.
Python bindings install into the existing venv with CUDA 13 CuPy <14 to retain NumPy
1.26 compatibility. Upstream's CUDA-12-specific Python dependency is bypassed with
explicit installation, not a source patch. Builds do not execute GPU tests.

This is an environment migration only: the older reconstruction paths documented
below still require pyCuSFM and must be replaced before they run in the new image.
The custom cuNLS local optimization diagnostic is implemented below; full pipeline migration remains pending.

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

The C++20 FFmpeg extractor decodes every frame by default, using NVDEC when
available. Unsupported hardware decoding falls back to CPU, restarting from the
beginning if a hardware error occurs mid-stream. A CUDA area-resampling kernel
reduces NV12, P010 and YUV420P surfaces before host transfer. Other surface formats
are downloaded and resized on CPU. CUDA allocations use the decoder's owning
context, stream and a reusable frame pool.

The resize kernel uses 32 × 4 thread blocks, with each warp accessing adjacent
scalar components (including interleaved UV). The normal path cooperatively loads
source tiles into shared memory and executes one unconditional block barrier
before consuming them. Partial edge blocks participate in that barrier before
inactive lanes return. Tiles are never overwritten within a block, so a second
barrier is unnecessary. Pixel layout and bit shifts are template specializations;
P010 samples are unpacked before integration and clamped/repacked at the output.
Ratios whose footprints exceed the 16 KiB shared-memory budget use direct reads.
This avoids oversized shared allocations; arbitrary-ratio shared gathers are not
assumed to be bank-conflict-free. All plane kernels use the decoder's stream,
with launch checks and one checked stream synchronization per frame.

CUDA changes must explain ownership, bounds, memory-access patterns and why each
barrier is needed. Warp synchronization alone is insufficient when data crosses
warps. Performance claims require target-device profiling, including register
pressure, occupancy, memory transactions and shared-bank conflicts, alongside
end-to-end extraction timing. Correctness validation should cover pitched planes,
partial tiles, NV12/P010/YUV420P, large ratios and comparison with a CPU area-resize
reference. Run Compute Sanitizer memory, race and synchronization checks before
treating a kernel change as validated. The present kernel changes have only been
reviewed statically; GPU correctness and performance remain unmeasured.

Full-video extraction detects visible CUDA devices. With multiple devices and a
usable packet index, it partitions the sequence near equally spaced frame counts
at keyframe timestamps. Each GPU owns an independent demuxer and decoder, seeks
back up to two earlier keyframes to establish reference state, and keeps only frames in its half-open timestamp
interval. The first section decodes from the beginning. Temporary per-section
outputs are joined before cleanup and validated against every indexed display packet PTS:
missing/duplicate frames, differing dimensions, or non-increasing times invalidate
the parallel result. Streams without a reliable one-packet/one-frame timestamp
index, or failed section validation, use sequential extraction. Explicit diagnostic
frame/time selections remain sequential to preserve their selection semantics.
This adds a compressed-packet scan but avoids decoding the entire video on every
GPU. It preserves all selected frames rather than sampling or dropping them.
Packets with `AV_PKT_FLAG_DISCARD` still feed the decoder but are not expected
display frames. Container `nb_frames` can count such preroll samples and is not
used as an exact display-frame count. Failed parallel validation retains packet
counts, section boundaries and the first mismatch in `parallel_diagnostic.json`,
copied into the completed run's `input_frames/` after successful sequential fallback.

Extraction produces working images with a maximum edge of 3 × VGGT's configured
resolution (1536 at the default 512). It never upscales or crops; even-dimension
rounding can slightly alter aspect ratio. RGB conversion and lossless PNG encoding
share a CPU budget of up to eight workers (at least one per GPU section), with at
most two outstanding frames per worker. A single progress bar counts successful
writes across sections, not queued work. Final file ordering follows timestamps,
independently of section completion order. Delayed decoded frames
are flushed at EOF; timestamps and original/working sizes are written to a manifest
only after all PNGs finish. The original video is retained for future full-resolution
stereo rendering. VGGT's own crop/resize remains authoritative for its predicted
intrinsics; working-image coordinates must not be mistaken for original-video pixels.

The Python adapter manages subprocess cancellation, progress and an extraction cache
under `.frames/` beside the source video. Cache keys include source path, size,
timestamps/inode, selection, resolution and executable hash. A per-key file lock
and atomic directory rename prevent partial or concurrent publication. Reuse checks
the manifest, expected files, sizes and timestamps; it is not a content checksum of
every image. Published runs hard-link immutable frames where possible, copying
across filesystems. Cache deletion does not invalidate saved run links. There is
no second Python decoding loop.
Native code uses namespaces, classes, explicit types, RAII resource ownership,
`[[nodiscard]]` where appropriate, and `this->` for instance member access.

VGGT inference uses the visible CUDA devices, estimating per-device capacity and
adapting overlapping section sizes from memory measurements. The target budget is
90% of each GPU's VRAM, accounting for other allocations; VRAM is not pooled and
90% utilization is not guaranteed. OOM retries reduce section sizes while retaining
all selected geometry frames. Video inputs intentionally reduce candidates to
visual keyframes before VGGT; `--all-frames` bypasses this for diagnostics.

Keyframe selection is a separate native pass over the decoded cache. The default
RaCo–ALIKED/LightGlue+ frontend runs in TensorRT, with bounded GPU extraction
workers and CUDA RANSAC. A single timestamp-ordered matcher/decision consumer spans
all decoder sections. Features are compared to the last accepted keyframe, then
verified against fundamental and homography models. RaCo–ALIKED/LightGlue+ is
the only supported keyframe frontend. The stronger model's support is checked against
minimum inliers, inlier ratio and coverage of a 4 × 4 grid in both views. Homography
support allows planar/rotation-dominated footage; image movement is not interpreted
as metric translation or guaranteed parallax. Repeated patterns and moving objects
can still produce incorrect geometrically consistent matches.

Sharpness and feature-count gates reject poor candidates. A keyframe is accepted
when inlier support falls relative to the best support since the current anchor,
or normalized image displacement grows sufficiently, subject to minimum spacing.
A buffered last reliable candidate bridges sudden loss; each subsequent match is
recomputed against the updated anchor. A sustained inability to connect frames is
reported as a tracking break and stops geometry. First/last usable views are retained;
fewer than three connected keyframes is an explicit diagnostic outcome. Thresholds
are initial tunable values in `configs/keyframes_raco.json`, not validated guarantees.

`python -m stereoforge.video.keyframe_demo --video VIDEO` runs selection alone
with an OpenCV desktop window. A main-thread observer displays the actual reference,
candidate, mutual descriptor matches, RANSAC mask, and decision without repeating
matching or changing selection order. Feature workers remain bounded while paused.
Space toggles playback, N steps, O toggles outliers, and Q/Esc saves an interrupted
diagnostic. The standalone native selector exposes the same window via
`--debug-view`. Debug reports remain separate from the production selection cache;
they include partial outcomes and do not require three selected frames to inspect.
No geometry models are loaded by this diagnostic command.

`RaCoALIKEDExtractor` and `LightGlueMatcher` implement the single keyframe path,
selected explicitly with `--keyframe-config configs/keyframes_raco.json` in
either demo. The implementation uses pinned
[LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX/tree/d12b4ba1632f558234e3f084e1f3d8bdf9147890)
models from the `third_party/lightglue-onnx` submodule with two separate static
ONNX exports. Docker copies the checked-out source, preserving the Git pin.
The matcher adapter invokes upstream's projection, attention and assignment
modules at fixed depth, then performs dense mutual-match filtering with the same
criterion as upstream. It retains one output slot per source keypoint instead of
constructing a variable-length match list. All model inputs and outputs have
fixed dimensions; exports containing NonZero are rejected.
Export uses upstream's Dynamo/opset-20 settings, with optimization disabled,
inline weights, and integer-division translation to avoid FP16 index overflow.
A post-export compatibility pass inlines local functions, folds constant
expressions and materializes Constant nodes as initializers. It rejects reduction
axes that remain nonconstant before TensorRT construction. ONNX validation runs
before cache publication and TensorRT construction. A new
export contract key prevents reuse of graphs from the legacy exporter.
The extractor uses upstream dense ranking with BatchNorm folded for inference,
and ALIKED's portable GridSample deformable-convolution path to avoid plugin
requirements. The model uses a fixed RGB canvas divisible by 32 and fixed top-K.
RaCo detection scores are spatial probabilities; the default detection cutoff is zero, while rank selection, border checks and match confidence remain active.
Python prepares and caches engines;
C++ performs frame inference using TensorRT 10.13.3.9.

RaCo–ALIKED takes RGB [0,1] images resized with preserved aspect ratio and
bottom/right padding to a fixed canvas. It outputs pixel keypoints, scores, and
128-dimensional descriptors. FP16 engine builds use FP16 floating-point I/O;
match indices use INT32. The fixed top-K slots remain aligned, while
weak detections and padding are excluded from feature counts and accepted matches.
LightGlue binds separate reference/candidate device buffers directly, then applies
`(xy - [canvas_width, canvas_height]/2) / (max(canvas_width, canvas_height)/2)` and pair assembly within the graph,
matching this upstream implementation.
The export retains the full fixed-depth network and learned assignment, returning
fixed-size mutual match indices and confidences. CUDA filters confidence, scores both fundamental-matrix and homography hypotheses,
and computes inlier ratio, coverage and median pixel displacement. The bounded
RANSAC budget is 2,000 hypotheses per family. Normalized DLT and rank-two
fundamental projection use double precision; consensus scoring uses pixel-space
errors. Rank-deficient samples are rejected and each family's winning model is
refitted on all its inliers, retaining the original if the refit scores worse.
The fundamental solver uses eight-point samples; results are not guaranteed
identical to OpenCV's CPU estimator. Correctness and performance need user validation. There is no claim
that learned matching eliminates false correspondences or repeated-structure ambiguity.

One bounded extraction worker runs per visible GPU when peer access to GPU 0 is
available; otherwise both stages use GPU 0. Acceptance remains sequential against
the last accepted reference. Shared frame ownership retains pooled device buffers,
which prevents reuse while queued or held as an anchor. Same-device descriptor
handoff binds the original RaCo–ALIKED output allocation; cross-device handoff
uses a cached peer mirror without host staging. Scores are retained alongside
points and descriptors. CUDA RANSAC consumes the matcher's original device outputs
on the same stream; only feature/selection aggregate statistics return to CPU.
The debug window additionally downloads points, matches and masks for display.
Input PNG decoding still occurs on CPU, with one pinned upload; CUDA converts
BGR to RGB and applies antialiased Keys bicubic interpolation (a=-0.5), with
half-pixel centers, clamped edges and weight normalization. Horizontal and vertical
passes use 32×8 blocks and shared-memory strips with unconditional barriers,
including partial edge blocks. Filter support widens by the downsampling ratio.
FP32 intermediate values retain negative cubic lobes; final pixels are clamped to
[0,1] before conversion to the model I/O precision. Each extraction worker retains
a reusable 3×source-height×resized-width FP32 device workspace. The second pass
writes directly into the zero-padded model canvas on the same CUDA stream.
Source dimensions are bounded to 65,536 and model canvases to 4,096 per side;
allocation failures remain explicit. Native binary hashes invalidate selections
after a resizer change. This resampler is not claimed bit-identical to an external
library; edge behavior, identity scaling, strong downscales, partial blocks and
FP16/FP32 accuracy need validation on the target GPU.
The GPU sharpness stencil uses a shared tile with halo, unconditional block
barriers, warp reductions and two global statistics atomics per block. This is
not a streaming NVDEC integration and not end-to-end zero-copy from video. The supplied config selects mixed FP16; FP32 remains selectable. TF32 stays disabled.
FP16 builds constrain reductions, softmax, normalization, elementwise and unary
floating-point arithmetic to FP32; external floating tensors use the selected
precision while indices stay INT32. This is not INT8
quantization, and matching accuracy must still be measured. Engine caches include
precision and runtime/GPU identities plus engine hashes. Build optimization level
is 5. Before each engine build, free and total VRAM are queried on that GPU.
The tactic budget is free VRAM minus max(10% of total VRAM, 1 GiB); workspace
receives half that allowance, leaving room for weights and build overhead.
Both limits are rounded down to powers of two and read back from TensorRT;
a rejected limit stops preparation instead of silently using a default.
Limits are recorded in each engine's `.build.json`; they do not impose a hard
process-wide CUDA cap or reserve memory upfront. Builder resources are released
before engine inspection. Existing sparse weights are enabled, available
legacy/edge-mask/JIT tactic sources are enabled, and the full runtime is used.
Tactic shared-memory limits retain the device maximum; no DLA fallback is needed.
The runner spin-waits on completion events and caches up to 32 graph executables
per context keyed by buffer addresses, with warm inference before capture and
ordinary enqueue fallback on unsupported capture. Models keep fixed shapes;
pooled output slots allow graph reuse without changing retained tensor ownership.
`python -m stereoforge.video.learned_models` prepares the models independently;
`--onnx-only` exports without a GPU, and `--precision fp32` builds a reference.
Missing upstream weights are downloaded and exported into our split ONNX contract.
A separate hashed ONNX cache survives engine-build failures and is reused across
precisions/devices. The CLI override does not change the demo configuration.
Models are prepared on first use or explicitly, not during Docker build.
The debugger identifies the frontend; thresholds still require footage-based
calibration. This backend does not change pyCuSFM's ALIKED configuration.

The selector saves acceptance reasons and
per-candidate metrics, with original displayed-frame indices/timestamps preserved
through renumbering into a run. Candidate PNGs remain cached; only selected frames
enter VGGT, reports and pyCuSFM. Keyframe selection does not replace ALIKED refinement.
Future full-rate stereo output must recover poses and suitable depth for intervening
frames; pose interpolation alone does not supply that geometry.

Section results are spooled to CPU/disk. Models are unloaded and CUDA caches
released before merging. The default overlap is 32 selected keyframes;
section sizes still adapt to each GPU's memory budget. Robust Sim(3) fits shared-frame depth geometry,
transforms cameras, and rescales depths. Earlier overlap predictions are retained
once. The first section sets the world/scale anchor. Alignment failure stops the
reconstruction; sequential drift is still possible. Alignment considers shared-camera
rotation/depth-scale seeds, per-frame point fits, and deterministic three-point
RANSAC hypotheses, then refits threshold inliers. Acceptance still requires median
depth-normalized point error at most 0.08, at least 50% point inliers at that
threshold, and median camera rotation disagreement at most 10 degrees. These are
heuristic quality checks, not guarantees of global accuracy or pixel reprojection bounds.
Before registration, every shared processed RGB tensor must match exactly. Independent
camera/depth transforms are compared by rotation, scale ratio, and their action at
a common 3D anchor (translation differences alone depend on the coordinate origin).
A majority must agree within 10 degrees, a 1.1 scale ratio, and 0.08 median-depth
units of anchor displacement. This consensus seeds Open3D tensor point-to-plane ICP
on shared-frame surfaces at three voxel resolutions. Scale stays fixed, Tukey
weights suppress outliers, and a weighted Jacobian conditioning check rejects
underconstrained surface updates. ICP results compete with the original candidates
under the original pixel and camera validation; ICP fitness alone cannot accept a merge.
ICP uses one CUDA device when available for downsampling, normals, and registration.
DLPack shares surface tensors with PyTorch for on-device conditioning reductions;
only a 6x6 matrix and transform/statistics return to CPU. Geometry preparation and
final acceptance checks remain on CPU. Automatic device selection warns on CPU
fallback; explicitly requested CUDA must be available. Runtime CUDA errors propagate.

Adjacent-section registration uses `boundary_alignment.py` to restrict all input
to identical shared-frame IDs, including RGB identity checks. At least five shared
frames must have sufficient jointly valid depth. Every third usable shared image
is held out; the remaining images alone seed Sim(3), consensus, and ICP. Every
held-out frame must pass the 8% depth-relative median error, 50% inlier support,
and 10-degree rotation-disagreement gates. On success the transform is applied
to the entire new section. Geometry outside the overlap is not independently
validated by this operation. Failed fitting or held-out validation requests boundary
recovery and retains diagnostics; it never searches for distant matching images.
Automatic boundary reinference is not implemented. Standalone alignment and
reprojection experiments and diagnostic multistart ICP have been removed.
On merge failure, all section tensor files survive staging cleanup under
`data/intermediate/failed_reconstructions/<id>/`, with `failure.json` containing
section boundaries, previous transforms, and available per-frame overlap diagnostics.
Processed RGB is embedded in the tensors; source image paths can refer to removed
staging files. There is no automatic resume command yet. Final arrays and compressed
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

- `input_frames/`: selected geometry images and a manifest with original indices,
  timestamps and keyframe-selection decisions for video inputs.
- `geometry.npz`, `metadata.json`, `run.json`: merged VGGT arrays and provenance.
- `previews/`, `contact_sheet.jpg`: source image/depth/confidence inspection.
- `index.html`: entry point to the final pyCuSFM viewer or diagnostic page.
- `pycusfm/input/`, `config/`, stage logs, and `workspace/`: native refinement inputs
  and diagnostics.
- `pycusfm/initialized_sparse/` and `initialized_binary/`: depth-seeded BA models.
- `pycusfm/workspace/ba_raw/`: unmodified optimized export.
- `pycusfm/workspace/sparse/`: validated sparse model.
- `pycusfm/initialization.json`, `refinement.json`, `comparison.json`, `index.html`:
  initialization counts, enabled stages, coverage, alignment and sparse viewer.

The self-contained WebGL viewer supports points, camera frustums, trajectory,
progressive playback and RGB/depth/confidence previews. Playback reveals a saved
final reconstruction; it is not live reconstruction. The final sparse viewer labels dense previews as VGGT references. Refined runs
do not generate a second VGGT 3D viewer or dense PLY. Dense arrays are retained
for recovery; explicitly disabling refinement still produces a VGGT-only viewer.
Native BA discards RGB colors, so validated points recover their measured seed
colors through unchanged observation associations, not by assuming point IDs
remain stable. Detailed native logs remain on disk while one shared progress
bar displays the active refinement stage.

Sparse cameras are aligned to VGGT with a similarity transform for comparison.
This removes global gauge differences, not local drift. Missing cameras are not
filled with invented refined poses. `complete` requires full frame coverage plus
aligned nonempty geometry; it does not certify reconstruction accuracy. Partial,
empty, failed and interrupted results retain diagnostic explanations.

The full runner preserves VGGT arrays and source previews if refinement fails. Recovery commands
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

## Historical: experimental hierarchical sparse reconstruction (removed)

The `stereoforge/reconstruction/` package separates reconstruction responsibilities:
`demo.py` handles the video, authorization-gated checkpoint resolution, request
identity and resume; `view_graph.py` builds tracks and the separator tree;
`inference.py` spools independent VGGT groups across visible GPUs; `triangulation.py`
initializes/rebuilds landmarks; and `pipeline.py` coordinates local and hierarchical
optimization and publishes the sparse viewer. `registration.py` estimates and
reconciles sparse Sim(3) candidates; `merge_validation.py` withholds overlap
observations and evaluates the optimized result. `refinement/bundle_adjustment.py`
contains the reusable standalone pyCuSFM BA adapter and sparse statistics.
Superseded two-section BA, camera-recovery and targeted-recovery modules are
removed. The working dense pipeline and saved-run refinement commands remain available;
obsolete alignment/reprojection experiment CLIs are removed. Saved outputs are not deleted.

The frontend matches temporal neighbors with the existing RaCo–ALIKED/LightGlue+
TensorRT implementation and CUDA F/H RANSAC. A verified pair requires at least
thirty inliers and a 25% inlier fraction. The native matcher exports detector
indices together with processed-image coordinates, retaining a bounded 32-frame
feature cache. Repeated feature IDs must have consistent pixel coordinates.
Confidence-ordered union-find rejects an edge that would introduce two observations
from one image, preserving the already-consistent components rather than deleting
them. These globally identified tracks are reused through every optimization stage.
There is no association between separate pyCuSFM ALIKED and RaCo feature sets here.

A balanced BFS ordering of the verified graph supplies candidate cuts. Boundary
cameras provide separators shared by both child groups; at least six are required,
with eight requested by default. Splits must make strict progress. This is a
separator hierarchy, not METIS nested dissection. The full input graph must be
connected; disconnected images are reported rather than silently removed. Groups
are capped at 32 images by default, independently of GPU memory capacity. VGGT
workers use separate GPUs when available, retain completed tensors on disk, and
exit before BA starts. One-GPU configurations use the same pipeline with one worker.

Verified tracks with three or more observations are robustly triangulated against
VGGT cameras. Candidate pairs require at least one degree of ray separation;
hypotheses are scored by positive-depth reprojection support and refitted from
inliers. This initial implementation uses triangulation rather than depth-averaged
landmark seeds. Local BA uses Cauchy loss and the native standalone policy. Every
parent estimates a provisional alignment from shared sparse landmarks: at least
six shared cameras and sixty unambiguous, depth-observable landmarks are required.
Eligibility is evaluated independently in each child before looking at cross-map
residuals: at least three observations, maximum observed-ray separation of 5–90°,
median reprojection error at most 1.5 pixels and maximum error at most three pixels.
The deterministic training/held-out split uses only eligible tracks. Both sets must
cover six shared cameras in each child, with at least three observations in three
cells of a 4×4 image grid per supported camera. Five degrees is an initial policy
motivated by the saved indoor overlap diagnostic, not a universal accuracy guarantee.
Weak tracks are reported separately and do not receive equal 3D trust. Training and
held-out Sim(3) landmark inlier fractions must each reach 80% within 5% of scene
depth. Every shared camera must agree within ten degrees and 5% of scene depth.
Landmark RANSAC initializes a seven-parameter Sim(3) optimization combining
reliable training-landmark coordinates with shared-camera centers and orientations.
Every third shared camera is held out from this objective (at least two), with
at least three remaining training cameras. Landmarks retain their separate held-out
split. Position residuals use the existing 5%-of-scene-distance normalization;
rotation residuals use ten degrees. Soft-L1 vector losses are averaged separately
for the three evidence groups and weighted equally, so track count cannot dominate
camera evidence. The SciPy trust-region solve adjusts log scale, rotation vector
and scene-normalized translation around the landmark centroid. Its scale multiplier
is bounded to [0.25, 4]; nonconvergence or an active scale bound rejects the candidate.
Camera matrices, intrinsics and section geometry are otherwise fixed. Existing
acceptance checks remain mandatory, including all shared cameras. Reports expose
initial/final objectives, training/held-out identities and convergence. This bounded
local solve diagnoses a candidate; failure is not a global infeasibility proof.

Cross-section pixel errors are recorded before BA but do not alone reject an
otherwise valid initialization. Reconciliation keeps one camera per image and
requires at least twenty joined tracks after fourteen-pixel seed filtering.
Before this filtering, weak tracks (joined or unpaired) are re-triangulated against
the reconciled cameras from their original surviving child observations. Robust
ray-pair hypotheses require at least one degree of parallax; retained tracks need
three positive-depth observations within four pixels and at least one degree of
parallax after filtering. Failed tracks are dropped with counts; their old 3D
coordinates are not retained as a fallback. Re-triangulation and global track
initialization share the same implementation.

Before joint BA, one measured overlap observation is withheld from eligible joined
tracks, leaving at least three training observations per track and ten per affected
camera. The remaining graph must be connected. At least forty observations total
and five per shared camera are required. The optimized model must preserve all
supported cameras, and at least 80% of withheld projections must be within five
pixels both overall and in every shared camera. Unknown/ambiguous optimized
landmark identities count as failures. Identity is recovered from surviving
measured observations, never native landmark numbering. Withheld observations
were used in child BA and initialization, and selection follows seed filtering;
this is not an independent evaluation of the whole pipeline. They remain excluded
from accepted intermediate tracks. Rejected candidates retain diagnostics and
viewers without a completion marker.
The same global tracks are retriangulated with the final merged cameras before a
last BA, allowing observations lost during local filtering to be reconsidered for
cameras still registered. Unregistered cameras are not silently reintroduced.

GTSfM-equivalent pose/calibration priors, GNC-TLS and final prior release are not
exposed/verified in this backend. Fixing the first camera does not establish metric
scale. Temporal candidate selection does not provide distant loop retrieval. Sparse
optimization never updates the original dense depth tensors. These are explicit
limitations rather than implicit claims of paper equivalence.

A run retains input identity, selection indices/timestamps, verified graph components,
global tracks, partition hierarchy, raw VGGT tensors, COLMAP seeds, optimizer logs,
per-node alignment and validation, final retriangulation and the colored sparse
viewer. `--resume` requires unchanged input/settings/checkpoint and verifies saved
track and hierarchy identity. It reuses saved matching, VGGT and leaf BA even when
merge code changes. Parent/final caches require the current policy, merge-code
fingerprint and child-result fingerprints. Older BA attempts are preserved under
`.previous_*` directories before retrying; unvalidated merges cannot be reused.
A partial output lists every missing camera and returns nonzero. Complete requires
all selected keyframes, not merely a connected subset. No inference or build was
performed as part of implementing this path; end-to-end quality remains unverified.

### Native source layout

Custom CUDA kernels and their launch implementations live in `.cu` source files
under `stereoforge/video/src/`. Kernel interfaces use `.cuh` under
`include/stereoforge/video/`. Host C++ uses `.cpp`/`.hpp` even when calling CUDA
runtime/driver or TensorRT APIs. The host `CudaResizer` class is declared in
`cuda_resize.hpp`; its kernels remain in `cuda_resize.cu`. Both language targets
require C++20. The decoder, keyframe selector
and pair-matching executable are active components of the supported pipelines.

### Future video-depth integration

Temporally consistent video depth is a future dense stage, not implemented by the
hierarchical sparse command. Align its scale (and any model-specific ambiguity)
to reliable reconstructed geometry, check depth against image correspondences and
occlusions, and retain confidence masks. Corrected sparse cameras do not by themselves
correct dense depths. A later depth-consistency/refinement stage must precede treating
those depths as geometrically consistent stereo-synthesis input. Do not assume
metric units without an external scale reference or verified metric-depth source.

### ACE0 research direction (not implemented)

[ACE0](https://nianticlabs.github.io/acezero/) alternates scene-coordinate regression
mapping and image relocalization, refining camera poses and calibration during
mapping. The [upstream partial-reconstruction workflow](https://github.com/nianticlabs/acezero#start-from-a-partial-reconstruction)
can train a scene model from posed images and use it to register additional images.
A possible StereoForge experiment is to initialize from a validated partial map and
re-estimate subsequent poses in that shared scene representation, instead of insisting
that independently optimized sections stay related by one rigid similarity.
This requires scene-specific neural training and a separate integration; pyCuSFM
bundle adjustment alone does not implement ACE0. Its calibration assumptions must
be reconciled with our per-image intrinsics before using exported poses. Reconstruction
quality and registration completeness would still need validation on the indoor clip.

### Bounded map-based boundary experiment

`reconstruction/boundary_demo.py` exercises registration in one coordinate system
using saved global feature tracks. The earlier child's cameras preceding the shared
interval seed the map; overlap cameras and observations are removed. Landmarks
retain at least three anchor observations. Their coordinates still originate from
previous BA involving overlap images, so the experiment is not an independent
reconstruction benchmark. No neighboring camera pose or landmark position seeds PnP;
only its per-frame calibration is reused.

Global-track association requires two matching measured observations and rejects
ambiguous point/track identities. CPU OpenCV EPNP/RANSAC estimates each target pose
without an initial pose guess; LM refines training inliers. A fixed quarter of
initial correspondences is held out for each target and remains excluded from all
later fitting. Forty correspondences, thirty training inliers, 70% support within
three pixels in both sets, and six cells of spatial coverage are required.

Successful registration extends existing tracks and triangulates new tracks using
the shared triangulator, then runs native pyCuSFM BA. Only the first camera is fixed
by the exposed backend; the whole anchor map is not locked, and metric scale is not
established. All previously supported cameras must survive and every registered
target must retain 70% held-out three-pixel support. Missing landmarks fail validation.
The command handles the overlap plus at most 32 additional frames, with a three-pass
limit and early stopping on no registration progress. It saves a separate colored
sparse viewer and detailed JSON/native artifacts without modifying the hierarchy.

## Historical: incremental reconstruction experiment (removed)

`frontend.py` owns shared processed-image preparation, native matching and viewer
export. `map_registration.py` contains the common PnP, track association, extension
and held-out validation used by both bounded and full-sequence reconstruction.
`incremental.py` coordinates seed creation and expansion; the default `demo.py`
selects this path for new videos. Legacy hierarchy orchestration remains separate.

The seed must retain all initial cameras with at least sixty points through BA.
Subsequent PINHOLE calibration starts from the median seed intrinsics. Incremental
registration optionally selects a joint pose/focal fit using only training observations:
RANSAC inliers drive robust least squares; all training observations score the candidate.
One focal multiplier is bounded to [0.9, 1.1]; aspect ratio and principal point remain
fixed. Nonconverged, poorly conditioned or bound-hitting fits retain the fixed-calibration
pose, as do fits that lose training inliers or fail to improve clipped training error.
Held-out observations never select the calibration hypothesis. This bounded correction
is not general self-calibration and does not establish that calibration caused a failure.
Changing image dimensions and substantial zoom remain outside this model. No later
VGGT camera pose is inserted. PnP retains the bounded experiment's 40-correspondence,
30-training-inlier, 70% three-pixel train/holdout and six-grid-cell requirements.
Held-out target observations remain excluded from PnP, triangulation and BA.

The map extends in temporal proximity order, capable of growing in either direction
from its registered frames. Training-qualified PnP proposals enter temporary maps for joint BA before held-out
acceptance; the 70% held-out threshold is unchanged. Held-out pixels remain excluded
from fitting, track extension and BA. Loss of any supported camera or failure of any
registered target's held-out check rejects the proposal, rolls back the entire map,
and allows other candidates to proceed. Native execution errors still stop the run.
Only validated proposals advance camera counts and the committed sparse-model pointer.
Per-attempt reports retain both pre-BA metrics and post-BA validation, including rejected
BA artifact paths. No-progress checkpoints preserve new holdout identities and attempt
history while retaining the previous validated map. The last validated sparse model and its holdout identities, calibration,
target list, track digest and report are referenced by an atomic state file. Native
failed/uncommitted rounds remain separate and are never reused as accepted state.
Resume restarts from that validated checkpoint. The checkpoint format stays compatible
with earlier incremental runs, preserving their held-out identities and map; registration
policy metadata records the new training-only calibration and per-camera BA behavior.
The bounded boundary demo retains its supplied per-frame calibration without this fit. Completion requires all selected
frames; no-progress runs publish partial coverage with missing IDs and nonzero exit.

The result is a colored sparse map and camera trajectory, not dense geometry or a
stereo video. The backend fixes only its first camera; metric scale and global drift
correction are not guaranteed. No new long-range loop matching is implemented.


## Custom cuNLS optimization (standalone diagnostic CLIs removed)

The saved-cluster diagnostic has been run successfully by the user, including
its pixel Jacobian check. The next implemented stage joins two overlapping
clusters; its runtime validation is pending. Neither command redirects the old
pyCuSFM end-to-end entry points.

- `optimization/include/stereoforge/optimization/`: host API `.hpp` and
  custom CUDA factor interface `.cuh`.
- `optimization/src/`: ordinary C++ graph/IO code and `.cu` factor kernels.
- `refinement/cunls.py`: typed options, normalization, native process lifecycle,
  output identity checks, filtering and diagnostics; no pyCuSFM dependency.
- `reconstruction/cunls_demo.py`: verified saved RGB/track coordinate grids,
  depth-initialized sparse cluster, before/after HTML reports.

The native pixel factor has three residuals: weighted horizontal/vertical pixel
error and an unweighted cheirality barrier. It uses cuNLS right-multiplicative
world-to-camera SE3 updates, Euclidean 3D points, and log(fx)/log(fy) states. All
11 tangent columns have analytic derivatives. Principal points are fixed and
lens distortion is outside this initial implementation. An optional runtime
central-difference check compares the CUDA Jacobian to an independent CPU FP64
reference of the right-SE3 update on eight observations.

For each measured track, initialization takes the coordinate-wise median of
VGGT depth-unprojected positions. Observations above 14 pixels are removed;
landmarks need three surviving views. This is a StereoForge initialization
choice, not a claim of an exact GTSfM reproduction. All requested cluster cameras
must be supported and at least 60 landmarks must survive initialization.

Before solving, subtract the first camera center and divide geometry/translations
by the median nonzero consecutive camera step. Priors default to 0.1 radians,
0.1 normalized translation units, and 10 focal pixels. The first camera remains
fixed; soft pose priors retain the seed scale. These are initialization priors,
not physical measurements, and should later be released in a correctly gauge-fixed
global stage. That global stage is not yet implemented.

GNC-TLS uses a 3-pixel cutoff, increasing mu by 1.6 between LM solves, with up to
64 rounds of 50 LM iterations. The two image residuals and Jacobians are multiplied
by sqrt(weight). Weights remain fixed within an LM solve. Landmarks with fewer
than three observations having sqrt(weight)>0.01 are held constant for that round,
preventing unsupported landmark states from making the system singular. Pose and
focal priors are not robustified. Each round records costs, support and frozen
landmarks. GNC termination requires stable truncated cost and no soft weights;
hitting the round limit is explicitly reported as a partial diagnostic.

CUDA uses bounded grid-stride launches with 128-thread blocks. Each observation
is independent, so no shared-memory tile or block barrier is required. Factor
work uses the cuNLS stream; synchronization occurs before CPU diagnostic reads.
This first subprocess/file interface copies data at the Python/native boundary;
it does not claim zero-copy or multi-GPU solving. A single visible device index
is selected, and the same path supports single-GPU machines.

Raw optimized models and filtered models are retained separately. Filtering uses
positive depth, at most three pixels reprojection error, and three views per point.
Complete diagnostic status also requires six observations per camera, connected
support, all original cameras, and the GNC convergence criterion. No fitting
statistic is presented as held-out validation or ground-truth accuracy.


### Historical two-cluster cuNLS experiment (CLI removed)

`python -m stereoforge.reconstruction.cunls_pair_demo --run RUN --device 0`
uses keyframe ranges [0,32) and [24,56) by default. The first saved VGGT seed is
reused when its indices match; missing clusters are inferred from the run's
validated input manifest and saved authorized checkpoint. Explicit `--reference`
and `--local` accept saved clusters from the same run instead. Inference workers
exit before BA starts. No decoding, feature extraction or matching is repeated.

Each cluster independently verifies its processed RGB against the feature grid,
initializes globally identified measured tracks from VGGT depth, and runs local
GNC-TLS cuNLS BA. Global JSONL track identities survive the native array interface
and filtering through `track_ids.json` and `retained_point_indices`. Shared-pixel
associations are checked against these identities before alignment.

The existing CPU registration implementation performs landmark RANSAC and a
seven-parameter camera-aware Sim(3) fit, with training/held-out landmark and camera
checks. Its depth-observability, coverage and acceptance gates remain unchanged.
A rejected alignment stops the experiment with a detailed `alignment.json`;
there is no forced merge, threshold relaxation or unrelated-frame search.

Reconciliation keeps one camera per global frame, joins unambiguous tracks and
retriangulates weak landmarks from measured rays. Joint BA withholds overlap
observations, then solves with custom cuNLS factors and unit observation weights
(`use_gnc=false`). This reuses the native C++/CUDA solver; no new upstream cuNLS
code is needed. The first camera stays fixed and soft pose/focal priors are taken
from the reconciled map. This is a bounded prior-regularized experiment, not the
future unanchored global optimization or an exact reproduction of GTSfM.

Success requires both local diagnostics to complete, accepted alignment, all
union cameras retained with connected support and six observations per camera,
and at least 80% of the withheld observations within five pixels overall and
in each shared camera. Missing optimized landmarks count as validation failures.
Held-out pixels were excluded from joint BA only: they participated in child BA
and initialization, so this does not establish independent scene accuracy.
Budget-limited joint LM is marked partial, even with a successful process exit.

Every invocation creates `RUN/cunls_pair_TIMESTAMP/` with local and combined
viewers, raw/filtered COLMAP models, solver logs, correspondence identities,
reconciliation details and validation results. Failure retains every artifact and
publishes the failing stage in the root report. The experiment covers only the
union of these two clusters (56 cameras by default), and never refines dense depth.


Joint-only retries use `reconstruction/cunls_joint_demo.py --pair PAIR
--lm-iterations 200`. The adapter forwards the original `joint_ba/input.json`
with only its iteration budget changed, preserving normalization, initialization
priors and measured pixels exactly. Validation reuses the saved excluded
observations and rejects policy mismatches or overlap with training pixels.
No upstream stages run. Fresh `joint_retry_TIMESTAMP` artifacts preserve the
previous result. A retry remains partial if either optimization completion or
pixel validation fails; thresholds are unchanged.


### Historical four-cluster cuNLS experiment (CLI removed)

`reconstruction/cunls_four_demo.py` runs a bounded balanced tree: locally refine
four 32-keyframe clusters at offsets 0,24,48,72, merge leaves 0+1 and 2+3, then
merge those accepted maps. The union contains 104 selected keyframes. Shared
`PairDiagnostic.merge` code implements each node; local solves retain GNC-TLS,
while joint solves now default to 200 LM iterations. Existing native C++/CUDA
factors are unchanged. Inference and optimization run sequentially on the chosen
device, including single-GPU systems.

Stable global track identities survive every merge and are used to fuse duplicate
landmarks even if local filtering removed their common pixels. Sim(3) estimation
still uses the existing shared-image correspondence and camera validation policy.
Earlier held-out observations are removed if another child would reintroduce them
into training. Root acceptance also reevaluates both child boundary holdouts;
missing landmarks count as failures. `ancestor_holdouts.json` preserves their
scope for joint-only retries, which repeat those checks as well.

A new `cunls_four_TIMESTAMP` contains `left`, `right`, and `root` merge reports
and viewers. The previous pair experiment is not modified or silently adopted:
local BA is recomputed, the existing first VGGT seed can be reused, and three
new clusters are inferred by default. The command does not rerun decoding or
matching. No parent consumes a rejected child. Full-sequence orchestration and
dense-depth refinement remain outside this experiment.


### Preserve child support and robustify joint BA

Current reconciliation preserves every established map landmark and its existing
observations before joint BA. Incoming observations of a shared global track are
proposals: admit them only within four pixels of the established point's projection
under the reconciled cameras. Conflicting tracks or rejected observations do not
replace or delete the established track. Shared tracks are not retriangulated at
this step. Incoming-only landmarks retain the existing support/reprojection checks.
Joint BA subsequently refines positions and cameras; its outlier filtering and all
current/ancestor holdout checks remain mandatory. Reconciliation reports proposed,
accepted and rejected observation counts separately from retained map landmarks.

New joint solves apply radial Huber loss to the two-dimensional pixel residual,
with delta=3 pixels. The CUDA factor represents rho(||e||²) exactly as the squared
norm of a transformed residual and differentiates the radial multiplier as well.
The cheirality barrier and pose/focal priors remain unrobustified. Local GNC-TLS
is unchanged, and combining it with Huber is rejected. Unweighted error reporting,
post-BA filtering and all withheld-observation acceptance thresholds are unchanged.
The optional finite-difference diagnostic also evaluates shifted observations to
exercise the Huber branch. These source changes have not been compiled or run by
the coding agent.

`validation_support.json` saves earlier-boundary unusable observation indices and
finite-error failure indices for each child, after reconciliation, after ancestor
holdout exclusion, in joint training, and after BA. Missing support remains a
validation failure. Saved-input retries preserve their original loss; a fresh
merge is required to recover landmarks discarded under the previous policy.

The optional pixel-Jacobian checker now uses an independent CPU FP64 residual
reference instead of perturbing and subtracting FP32 GPU residuals. It implements
right-multiplicative SE3 coordinate perturbations, projection and radial Huber
loss, and compares against the production CUDA analytic Jacobian. Unperturbed
CUDA/reference residual values are also compared. Finite differences refine from
0.001 through twelve halvings/levels, requiring two consecutive consistent passing
estimates. Original derivative tolerances remain unchanged. Production BA retains
its CUDA FP32 factors; FP64 is only for the optional diagnostic. Failure records
all estimates and still stops the solve.
