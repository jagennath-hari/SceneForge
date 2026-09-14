# StereoForge

Geometry-aware stereo video synthesis, starting with **one continuous, uncut
recording** such as drone footage or a walkthrough.

The implemented pipeline is:

```text
Video → VGGT-Ω dense geometry → ALIKED/LightGlue tracks
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
Native C++20 video extraction is built in the geometry image. Dependency or native
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

Every frame is retained by default. VGGT distributes overlapping sections across
visible GPUs with adaptive memory budgets, then unloads its models before CPU
merging. ALIKED refinement follows. Set `refinement.enabled: false` in
`configs/default.yaml` for VGGT only. ALIKED is the only supported feature family.

Optional diagnostic flags include `--frames`, `--start-seconds`, `--duration`,
`--device cuda:0`, `--output`, and `--debug`. Frame/time selections are explicit;
they are not required for full-video processing. Only use `--meters-per-unit`
when scale is independently calibrated.

## Inspect results

Open the new host-side `data/intermediate/geometry_<timestamp>/index.html`.
The self-contained WebGL viewer displays VGGT points, trajectory, camera frustums,
RGB/depth/confidence previews and progressive playback. Playback uses the saved
reconstruction; it is not a live inference display.

Follow **View pyCuSFM reconstruction** for the validated sparse result or a
failure report. VGGT provides dense depth; pyCuSFM provides sparse landmarks and
optimized cameras. The sparse viewer's depth previews remain VGGT references.
Missing refined cameras are reported explicitly and are not filled with VGGT
poses labeled as refined.

Useful artifacts inside each run:

| Path | Contents |
|---|---|
| `geometry.npz`, `metadata.json`, `run.json` | Dense arrays, camera data and provenance |
| `previews/`, `contact_sheet.jpg`, `point_cloud.ply` | Dense inspection artifacts |
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
directly; previous results remain intact. The validator restores filenames from
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
