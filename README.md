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

Defaults: 64 keyframes per VGGT-Ω window, 32 shared keyframes, 1,000 joint BA
iterations, loop retrieval and Rerun enabled. Window dimensions count selected keyframes, not
video frames. To override them:

```bash
python -m stereoforge.reconstruction \
  --video data/input/indoor_travel.MP4 \
  --window-size 64 --overlap 32 --lm-iterations 1000
```

Use `--no-rerun` for headless processing, `--device cuda:0` to select one GPU,
`--debug` for exception tracebacks, and `--diagnostics` for CUDA Jacobian checks.
`--device cuda` distributes VGGT-Ω windows across visible GPUs; BA uses the first.
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

Input manifests, feature matches, tracks and raw VGGT-Ω windows also live inside
the run directory. These are needed for inspection and reuse; there is no separate
`data/intermediate` output location. Replay a recording with `rerun PATH/pipeline.rrd`.

To reuse a run's unchanged inputs and VGGT-Ω windows in a new map-building attempt:

```bash
python -m stereoforge.reconstruction --resume data/output/reconstruction_TIMESTAMP
```

Resume checks the video, checkpoint, settings and implementation identity. After
code changes, start a fresh `--video` run; matching keyframe/model caches can still
be reused. Existing saved runs are never relocated by a new invocation.

## Source layout

- `stereoforge/video/`: Python model/cache adapters and native decoding, keyframes and matching.
- `stereoforge/geometry/`: VGGT-Ω inference, tensor types, checkpoint loading and storage.
- `stereoforge/reconstruction/`: orchestration, feature tracks, sparse snapshots and dense refinement.
- `stereoforge/optimization/`: Eigen map building and CUDA/cuNLS bundle adjustment, exposed through pybind11.
- `stereoforge/visualization/`: Rust Rerun SDK adapter.
- `stereoforge/utils/`: shared progress, validation and export helpers.
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

See [the architecture specification](StereoForge%20Architecture%20Specification.md)
for the implementation and coordinate conventions.
