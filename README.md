# StereoForge

StereoForge reconstructs a continuous video into an optimized camera trajectory,
colored sparse landmarks and a refined dense point cloud, with live visualization
in Rerun.

```text
Video → streaming keyframe selection → measured feature tracks
      → overlapping VGGT-Ω windows → local BA + Sim(3) common map
      → shared-calibration global BA → dense refinement and fusion
```

## Environment

On the host:

```bash
git submodule update --init --recursive
bash scripts/build_and_start.sh
```

The script uses BuildKit to build CUDA → base → geometry → cuNLS → Rust/Rerun,
then opens a shell as the host user. A running `stereoforge` container is attached
without rebuilding. Stop it before rebuilding Dockerfiles, C++/CUDA or Rust code.
Python source, configuration, data and caches have individual bind mounts.

The environment requires an NVIDIA GPU and NVIDIA Container Toolkit. It uses
CUDA 13.2.1, the pinned PyTorch CUDA 13.2 wheels and TensorRT 10.13.3.9. Our native
backend links directly to cuNLS; the Python cuNLS package is not needed. The launch
script supplies NVIDIA runtime, host IPC/network/PID, privileged mode and X11 access.

Request access to VGGT-Ω on Hugging Face, then place your authorized read token
in `.secrets/hf_token` before starting the container. This ignored file is mounted
read-only at runtime; credentials are not baked into an image. Model weights and
TensorRT engines are downloaded or prepared automatically when absent and retained
in the mounted `.cache` directory.

## Run

Inside the container:

```bash
python -m stereoforge.reconstruction --video data/input/barn.mp4
```

Defaults: 64 keyframes per VGGT window, 32 shared keyframes, 1,000 joint BA
iterations and Rerun enabled. Window dimensions count selected keyframes, not
video frames. To override them:

```bash
python -m stereoforge.reconstruction \
  --video data/input/indoor_travel.MP4 \
  --window-size 64 --overlap 32 --lm-iterations 1000
```

Use `--no-rerun` for headless processing, `--device cuda:0` to select one GPU,
`--debug` for exception tracebacks, and `--diagnostics` for CUDA Jacobian checks.
`--device cuda` distributes VGGT windows across visible GPUs; BA uses the first.
Keyframe selection settings are in `configs/keyframes_raco.json`.

Decoding and keyframe selection run together. Only accepted keyframes are saved
as images; rejected frames retain timestamp metadata. The source video must stay
available for connection repair. Persistent `.keyframes` caches beside the video
avoid repeating selection when inputs and settings match.

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

Input manifests, feature matches, tracks and raw VGGT windows also live inside
the run directory. These are needed for inspection and reuse; there is no separate
`data/intermediate` output location. Replay a recording with `rerun PATH/pipeline.rrd`.

To reuse a run's unchanged inputs and VGGT windows in a new map-building attempt:

```bash
python -m stereoforge.reconstruction --resume data/output/reconstruction_TIMESTAMP
```

Resume checks the video, checkpoint, settings and implementation identity. After
code changes, start a fresh `--video` run; matching keyframe/model caches can still
be reused. Existing saved runs are never relocated by a new invocation.

## Source layout

- `stereoforge/video/`: Python model/cache adapters and native decoding, keyframes and matching.
- `stereoforge/geometry/`: VGGT inference, tensor types, checkpoint loading and storage.
- `stereoforge/reconstruction/`: orchestration, feature tracks, sparse snapshots and dense refinement.
- `stereoforge/optimization/`: Eigen map building and CUDA/cuNLS bundle adjustment, exposed through pybind11.
- `stereoforge/visualization/`: Rust Rerun SDK adapter.
- `stereoforge/utils/`: shared progress, validation and export helpers.
- `docker/`, `scripts/`, `configs/`: environment and active keyframe configuration.

The current pipeline verifies temporal image pairs. VPR retrieval and loop closure
are not integrated yet. Stereo synthesis is outside the current scope.
Reconstruction units are not established meters, and a complete run does not
establish geometric accuracy. Unresolved regions are reported as partial output.

See [the architecture specification](StereoForge%20Architecture%20Specification.md)
for the implementation and coordinate conventions.
