# SceneForge: GPU-Accelerated 3D Reconstruction from Monocular Video

SceneForge reconstructs continuous monocular video into a shared 3D map by combining
learned keyframe selection, VGGT-Ω geometry priors, and GPU-accelerated bundle
adjustment. The system aligns overlapping reconstructions, jointly refines camera
poses, calibration, and landmarks using temporal and verified loop correspondences,
and fuses refined depth into a colored dense point cloud. Live Rerun visualization
reveals the reconstruction as it develops, from selected keyframes to the final map.

## 🖥️ Tested Configuration

SceneForge has been tested on:

- 🐧 **Ubuntu:** 24.04
- 🧠 **GPUs:** 2 × NVIDIA RTX A6000, with 48 GB VRAM per GPU
- ⚙️ **CUDA (host):** 13.3
- 🧊 **Environment:** Docker with NVIDIA Container Toolkit and GPU support

> This is the tested reference configuration. Memory requirements depend on image
> resolution and the number of keyframes per VGGT-Ω window.

## Pipeline

```text
Video → streaming keyframe selection → measured feature tracks
      → overlapping VGGT-Ω windows → local BA + Sim(3) common map
      → shared-calibration global BA → dense refinement and fusion
```

## Environment

On the host:

```bash
git submodule update --init --recursive
bash scripts/build_and_start.sh data/input/barn.mp4
```

Pass one video path, relative to your current directory or absolute (including
videos outside this repository). The host needs FFmpeg's `ffprobe`: the script
checks for a readable video stream with at least two decodable frames before
building or starting Docker. It rejects missing files, still images and non-video
files regardless of their extension. This checks the beginning of the video;
damage later in the recording can still cause decoding to fail.

The script uses BuildKit to build CUDA → base → geometry → cuNLS → Rust/Rerun,
then directly runs reconstruction as the host user. It exits with the pipeline's
status and removes the container when processing finishes. If a `sceneforge`
container already exists, stop/remove it before launching another run.
Python source, configuration, data and caches have individual bind mounts. The
selected video is mounted read-only at `/input/video`, without copying it or
mounting its parent directory.

The Python package and Docker container are named `sceneforge`; the container
workspace is `/workspace/SceneForge`. After the project rename, rebuild the images
and use `python -m sceneforge.reconstruction`. Saved runs retain their historical
paths and implementation signatures, so start a new `--video` run rather than
resuming a run produced before the rename.

The environment requires an NVIDIA GPU and NVIDIA Container Toolkit. It uses
CUDA 13.2.1, the pinned PyTorch CUDA 13.2 wheels and TensorRT 10.13.3.9. Our native
backend links directly to cuNLS; the Python cuNLS package is not needed. The launch
script supplies NVIDIA runtime, host IPC/network/PID, privileged mode and X11 access.

The geometry image builds xFormers v0.0.34 from pinned source against the installed
PyTorch/CUDA stack, using Ninja and `MAX_JOBS=$(nproc)`. It builds CUTLASS attention
for SelaVPR++ (including FP32); the separate Flash Attention build is disabled.
`XFORMERS_CUDA_ARCH_LIST` in `docker/Dockerfile.geometry` includes the RTX A6000's
SM 8.6 and other supported targets, so Docker builds do not require a visible GPU.
After rebuilding, inspect the available operators inside the container with
`python -m xformers.info`. Source-build compatibility still needs verification
with the actual image build.

Request access to `facebook/VGGT-Omega` on Hugging Face and wait for approval,
then place a read token from that account in `.secrets/hf_token`. The token must
permit downloads from that repository. The launcher stops before Docker if the
file is missing, unreadable, empty or whitespace-only, and prints instructions to
create it. File presence does not prove access: Hugging Face checks authorization
when the checkpoint needs downloading. This ignored file is mounted read-only at
runtime; credentials are not baked into an image. Model weights and
TensorRT engines are downloaded or prepared automatically when absent and retained
in the mounted `.cache` directory.

## Run

From the host, one command builds the environment and runs the pipeline:

```bash
bash scripts/build_and_start.sh "/path/to/my video.mp4"
```

Outputs are saved in this repository's `data/output/`. The script launches Rerun
using the existing pipeline defaults. If you already have a container shell, the
Python entry point remains available for custom options:

```bash
python -m sceneforge.reconstruction --video data/input/barn.mp4
```

Defaults: 64 keyframes per VGGT-Ω window, 32 shared keyframes, 1,000 joint BA
iterations, loop retrieval and Rerun enabled. Window dimensions count selected keyframes, not
video frames. To override them:

```bash
python -m sceneforge.reconstruction \
  --video data/input/indoor_travel.MP4 \
  --window-size 64 --overlap 32 --lm-iterations 1000
```

Use `--no-rerun` for headless processing, `--device cuda:0` to select one GPU,
`--debug` for exception tracebacks, and `--diagnostics` for CUDA Jacobian checks.
`--device cuda` distributes VGGT-Ω windows across visible GPUs; BA uses the first.
Keyframe selection settings are in `configs/keyframes_raco.json`.

Decoding and keyframe selection run together. Only accepted keyframes are saved
as images; rejected frames retain timestamp metadata. The source video must stay
available for connection repair. The launcher persists keyframes in
`.cache/sceneforge/video-keyframes/`, mounted at `/input/.keyframes`, so it never
writes beside your original video. Direct Python invocations use `.keyframes`
beside their input video. Caches avoid repeating selection when inputs and settings
match; the launcher's new input path produces a different cache identity from
previous direct Python runs.

## Output

Each new run is saved under `data/output/reconstruction_TIMESTAMP/`.
`--output PATH` selects a new, unused directory instead. Data and generated output
are ignored by Git.

| Artifact | Contents |
| --- | --- |
| `status.json` | Completion state, camera coverage, warnings and stage timings |
| `trajectory.json` | Camera-to-world poses, processed-image intrinsics and timestamps |
| `point_cloud.ply` | Dense result when refinement succeeds; otherwise the sparse map |
| `sparse_point_cloud.ply` | BA landmarks, preserved after dense export |
| `dense_point_cloud.ply` | Fused dense geometry |
| `dense/` | Refined depth, support masks and previews |
| `map_TIMESTAMP/pipeline.rrd` | Rerun recording for the map-building attempt |

Input manifests, feature matches, tracks and raw VGGT-Ω windows also live inside
the run directory. These are needed for inspection and reuse; there is no separate
`data/intermediate` output location. Replay a recording with `rerun PATH/pipeline.rrd`.

To reuse a run's unchanged inputs and VGGT-Ω windows in a new map-building attempt:

```bash
python -m sceneforge.reconstruction --resume data/output/reconstruction_TIMESTAMP
```

Resume checks the video, checkpoint, settings and implementation identity. After
code changes, start a fresh `--video` run; matching keyframe/model caches can still
be reused. Existing saved runs are never relocated by a new invocation.

## Source layout

- `sceneforge/video/`: Python model/cache adapters and native decoding, keyframes and matching.
- `sceneforge/geometry/`: VGGT-Ω inference, tensor types, checkpoint loading and storage.
- `sceneforge/reconstruction/`: orchestration, feature tracks, sparse snapshots and dense refinement.
- `sceneforge/optimization/`: Eigen map building and CUDA/cuNLS bundle adjustment, exposed through pybind11.
- `sceneforge/visualization/`: Rust Rerun SDK adapter.
- `sceneforge/utils/`: shared progress, validation and export helpers.
- `docker/`, `scripts/`, `configs/`: environment and active keyframe configuration.

## Global descriptors and loop closure

During **Preparing feature images**, two CPU workers prepare geometry images and
SelaVPR++ inputs from the same decoded RGB. GPU batches compute DINOv2-base/GeM
2,048-dimensional descriptors (`hashing=False`, `rerank=False`). Retrieval uses
upstream's normalized 322×322 input; geometry retains its existing VGGT-Ω grid.
The local submodule supplies model code, and weights download into the mounted
Torch cache on first use. Descriptor caches include image contents, checkpoint,
model implementation and preprocessing identity.

Each keyframe proposes up to three distant candidates, excluding frames within
32 keyframes or less than ten seconds. Candidates for one query are spread across
temporal regions at least five seconds apart. The deduplicated proposals join
temporal pairs in **Verifying image pairs**. RaCo–ALIKED/LightGlue+ and RANSAC run
once per requested pair; local features are reused within each native batch.

Loop pairs require at least 30 RANSAC inliers, 25% inlier ratio, coverage of four
cells in each image's 4×4 grid, and a second verified pair within two keyframes on
both sides. Consistent loop observations enter the same measured feature tracks
and landmark-based BA as temporal observations. This is not a separate pose graph.
Connection repair counts temporal-only tracks so loops cannot hide a weak local
boundary. Repair frames reuse descriptors and measurements wherever possible.

`status.json` reports candidates, geometric verification, corroborated pairs,
consistent loop tracks, and loop landmarks/pairs/observations retained in the final
map. These counts differ: retrieval does not guarantee usable BA constraints.
`graph.json`, `loop_tracks.json`, and the repair round's `matching/loop_verification.json`
retain the evidence. A geometry-verified pair without neighboring support is not
added to the tracks.

Rebuild Docker to include SelaVPR++ and its FAISS import dependency, then use the
same command. `--no-loop-closure` provides a temporal-only comparison. Start a fresh
`--video` run after this change; older reconstruction policies cannot resume, but
video/model caches remain usable. Descriptor inference and extra verification add
computation even though no separate image-preprocessing stage is introduced.

Stereo synthesis is outside the current scope.
Reconstruction units are not established meters, and a complete run does not
establish geometric accuracy. Unresolved regions are reported as partial output.

See [the architecture specification](SceneForge%20Architecture%20Specification.md)
for the implementation and coordinate conventions.
