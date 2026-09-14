# StereoForge

VGGT-Ω geometry → adaptive stereo baseline → StereoSpace right-eye synthesis →
side-by-side video. pyCuSFM provides optional geometric refinement.

Current scope: **one continuous, uncut recording**, such as drone footage or a
walkthrough. Shot detection and edited movies are outside this first version.

Run `bash scripts/build_and_start.sh` to build and enter the environment.

All Python dependencies share `/opt/stereoforge-venv`, created by uv using
Ubuntu's Python and already on `PATH`. uv from the `latest` image installs the Python packages;
the legacy TensorRT components retain their working pip installation in the same
environment. Both installers use `docker/constraints.txt`. BuildKit caches uv
and pip downloads, and installed files are independent of those caches.

Add dependency changes to the Dockerfiles and rebuild. This uses `uv pip install`,
not `uv sync`; a full dependency lockfile has not been generated.

## First geometry result

For file-based Hugging Face credentials, create `.secrets/hf_token` on the host
and paste only your read-only HF token into it (one line, no quotes or `HF_TOKEN=`).
From the host repository, create the private file and open it in your editor:

```bash
mkdir -p .secrets
chmod 700 .secrets
touch .secrets/hf_token
chmod 600 .secrets/hf_token
nano .secrets/hf_token
```

The startup script optionally mounts this file read-only at `/run/secrets/hf_token`
and sets `HF_TOKEN_PATH` to that path. This is a plain Docker bind mount, not a
Swarm secret. `.secrets/` is excluded from Git and the Docker build context.
The token stays in a plaintext host file, accessible to your user and root;
it is not embedded in an image or exposed as a Docker environment-variable value.

After creating or replacing the file, stop the existing container and run
`bash scripts/build_and_start.sh` again so the new mount is used. Then use
`--download-checkpoint` normally; no `hf auth login` is needed. Avoid running
`hf auth login/logout` with the read-only token mount; edit/remove the host file
and recreate the container instead. Revoking the token in HF settings invalidates
it. Cached model downloads still persist independently in `.cache/`.

If `.secrets/hf_token` is absent, the interactive login method below still works.

The demo accepts an ordered folder of images from **one shot**, or an entire
continuous recording. Every frame is processed by default using overlapping
sections distributed across visible GPUs. Inputs must contain no cuts. Code and
configuration are mounted into the running container, so these changes need no
image rebuild.

Obtain access to the [VGGT-Omega checkpoints](https://huggingface.co/facebook/VGGT-Omega).
Use the non-text-aligned `vggt_omega_1b_512.pt` checkpoint. Either put it in
`weights/` on the **host**, or run `hf auth login` inside Docker and pass
`--download-checkpoint` to the demo. The latter downloads into the persistent HF
cache; it does not write to the read-only `weights/` mount.

For credential entry, create a read-only token in your
[Hugging Face token settings](https://huggingface.co/settings/tokens), then run
this inside Docker:

```bash
(
  umask 077
  mkdir -p "$HF_HOME"
  chmod 700 "$HF_HOME"
  hf auth login
)
```

Paste the token only at the CLI's hidden prompt. Answer **No** if asked to add it
as a Git credential. Do not put the token in a command argument, Dockerfile,
build argument, source file, or chat. Login stores the token locally in plain
text under `HF_HOME`; the private directory restricts access by other ordinary
host users. It is not encrypted and remains accessible to your user and root.
The existing `.cache/` mount persists login across container restarts and is
excluded from Git and the Docker build context. Treat it as private when backing
up or sharing the workspace. The startup script also restricts this directory.

The demo's `--download-checkpoint` uses the saved login automatically and reuses
the cached checkpoint on later runs. Run `hf auth logout` to remove saved login.
See the [Hugging Face authentication guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication).

For a folder of PNG/JPEG images in `data/input/frames/`, run inside
Docker from `/workspace/StereoForge`:

```bash
python -m stereoforge.geometry.demo \
  --images data/input/frames \
  --download-checkpoint
```

Images are sorted naturally (`frame_2` before `frame_10`), and all are used.
All input frames must have the same dimensions.

Our sample is `data/input/forest_road.mp4`, copied from the pinned VGGT-Omega
submodule's `examples/forest_road.mp4`. It is a 30.45-second, 1280×720 aerial
forest-road clip at approximately 23.976 FPS. Sampled frames across the clip show
the same continuous scene. Big Buck Bunny has been removed from `data/input/`.

This command processes the whole video, using every decoded frame:

```bash
python -m stereoforge.geometry.demo \
  --video data/input/forest_road.mp4 \
  --download-checkpoint
```

The demo records timestamps and retains every frame (`geometry.max_frames: null`).
It discovers GPU count and each device's free/total VRAM, then assigns overlapping
sections using a conservative starting-size estimate. GPUs hold separate model
copies; their VRAM is not pooled. On CUDA out-of-memory, the worker splits the
failed section again with overlap and uses smaller sections for subsequent work.
Successful sections are temporarily saved to disk and merged in temporal order.

Defaults in `configs/default.yaml` are at most 64 frames per initial section,
8 shared frames, and a memory estimate using 80% of currently free VRAM with a
model reserve. This is a heuristic, not a guaranteed memory bound. Retries stop
with an error if even 9 frames cannot fit with the default overlap. `--device cuda`
uses all visible GPUs; `--device cuda:0` selects one. The CPU path uses one worker.

The merger fits scale, rotation and translation using confidence-filtered matching
pixels in shared frames. Depth and camera poses are transformed together; shared
frames retain the earlier reconstruction. The first section anchors world
coordinates and scale. Poor alignment stops the run rather than publishing a
disconnected reconstruction. `run.json` records device memory snapshots, OOM
retries, section ranges and alignment transforms/errors. Inspect these and the
camera trajectory: sequential alignment can accumulate drift or leave seams;
there is no global optimization or loop closure yet. Moving subjects can also
produce inconsistent geometry. `--meters-per-unit`, if supplied, calibrates the
first section's reconstruction scale after alignment.

Final merging and report export still need host RAM proportional to the video,
plus disk space for extracted frames and temporary predictions.

Terminal progress shows decoded/saved frames, per-GPU completed sections, merging
and preview generation. Per-GPU status distinguishes model loading, preprocessing,
inference and saving. Section totals increase when an OOM retry splits work.
Elapsed time refreshes during blocking operations; it is not a measurement of
within-section GPU completion. Decode totals come from video metadata when present;
unknown totals show a count without a percentage. Redirected output uses periodic
status logs instead of animated bars. Detailed per-frame summaries use `--debug`.
`tqdm` is explicitly included in the base Dockerfile; if an older environment
lacks it, install it inside Docker with `uv pip install tqdm`.

`--frames`, `--duration` and
`--start-seconds` remain optional diagnostic controls: frames alone takes the
first N frames, duration alone keeps all frames in that interval, and combining
frames with duration samples across the interval. None is needed for a full run.

If the checkpoint is already under host `weights/`, omit `--download-checkpoint`.
Use `--checkpoint PATH` for another local checkpoint and match `--resolution` to
its training resolution (512 by default). Run `--help` for other options.
Upstream controls mixed precision;
its geometry heads output float32.

Each run creates a new directory under `data/intermediate/geometry_<timestamp>/`:

| Artifact | Contents |
| --- | --- |
| `index.html` | Offline frame slider, RGB/depth/confidence panels, rotatable point cloud and camera trajectory |
| `contact_sheet.jpg` | Overview of up to 24 evenly spaced frames with depth and confidence previews |
| `previews/` | Processed RGB, depth and confidence PNGs for every frame |
| `geometry.npz` | Raw depth/confidence, valid masks, K, camera-to-world, processed RGB, indices and units |
| `metadata.json` | Per-frame statistics, camera matrices, dimensions, confidence thresholds and visualization scales |
| `point_cloud.ply` | Subsampled colored world-space points for external 3D viewers |
| `run.json` | Checkpoint/configuration provenance, frame count and decoded video timestamps |
| `input_frames/` | Extracted source PNGs when using `--video` |

Open `data/intermediate/geometry_<timestamp>/index.html` from the **host repository**
in your browser. No server or network connection is needed. Drag to rotate and
scroll to zoom. Depth previews use one sequence-wide color range: yellow is near,
purple is far, and black is invalid or filtered. Confidence has a separate shared
color range with yellow meaning high confidence. The viewer is a diagnostic
preview, not a fused reconstruction or a stereoscopic output.

Array conventions:

- `depth`, `confidence`, `valid_mask`: `[N,H,W]` on the **processed** image grid.
- `processed_rgb`: `[N,H,W,3]` uint8 RGB.
- `intrinsics`: `[N,3,3]`, in processed-image pixels. Upstream can crop/resize;
  these intrinsics must not be applied directly to the original image dimensions.
- `camera_to_world`: `[N,4,4]`, converted from upstream world-to-camera poses.
  Camera axes are X right, Y down, Z forward; depth is positive camera Z.
- Confidence preserves upstream's `1 + exp(logit)` scores, not probabilities.
  By default, the lowest 20% of otherwise-valid scores in each frame are filtered
  for previews and statistics. Raw predictions in the NPZ remain unchanged.
- Depth and translations are labeled `reconstruction_units` by default. Supply
  `--meters-per-unit` only with an independently known calibration; it scales
  both depth and camera translations. Separate inference runs are not aligned.

`data/`, `weights/`, caches and Python bytecode are ignored by Git; `.gitkeep`
files remain trackable. No inference, checkpoint download, or tests were run while
implementing this milestone; the commands above are for validation in your container.

## Python structure

The CLI command and output filenames remain the same after the refactor.
Responsibilities are separated into a small set of typed components:

| Component | Responsibility |
| --- | --- |
| `GeometryConfig`, `PreviewConfig`, `DemoConfig` | Frozen, validated settings; reject malformed YAML and unknown geometry/preview keys |
| `DemoRequest`, `GeometryDemoRunner` | Validate a run, resolve a checkpoint, coordinate inference and publish outputs |
| `VideoFrameSampler` | Decode every video frame by default and preserve actual, increasing timestamps |
| `VGGTOmegaGeometryEstimator` | Lazy model loading, sequence inference, camera conversion and model cleanup |
| `AdaptiveGeometryEstimator` | GPU discovery, overlapping sections, smaller OOM retries and ordered merging |
| `SimilarityTransform`, `align_overlap` | Robust shared-pixel registration and consistent depth/pose transforms |
| `FrameGeometry`, `GeometrySequence` | Tensor shapes, coordinate conventions, units and source metadata |
| `GeometryReportWriter` | Export arrays, image previews, sampled points and the offline HTML report |

The tensor math remains in typed functions. The HTML viewer is a separate
template under `stereoforge/utils/templates/`. No new Python dependencies were
added. Frozen dataclasses prevent field reassignment; tensor contents remain
mutable and should be treated as read-only by consumers.

Runs are written into a temporary sibling directory and published by rename only
after the report completes. Failed/interrupted runs clean up their temporary
outputs; existing nonempty output directories are never overwritten. Checkpoint
caches are independent and remain available for retries. Add `--debug` for a
traceback; ordinary errors report actionable messages, including GPU memory
exhaustion and denied checkpoint access.

The same flow is usable from Python without argument parsing:

```python
from pathlib import Path

from stereoforge.geometry.config import DemoConfig
from stereoforge.geometry.runner import DemoRequest, GeometryDemoRunner

config = DemoConfig.from_yaml(Path("configs/default.yaml"))
request = DemoRequest(
    images=Path("data/input/frames"),
    checkpoint=Path("weights/vggt_omega_1b_512.pt"),
    output=Path("data/intermediate/geometry_example"),
)
result = GeometryDemoRunner(config).run(request)
print(result.output / "index.html")
```

The full-video changes have not been executed or tested by the assistant.
Run the command above to validate them in your container.

Next milestone: inspect the full sequence's geometry, then implement adaptive
baseline control and temporal smoothing, followed by StereoSpace and SBS encoding.
Adaptive sections and alignment now provide the full-video path; their quality
must be validated on your footage. Shot detection is outside this scope.
pyCuSFM remains disabled until the core stereo synthesis pipeline works.
