# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jagennath Hari
#
# This file is part of SceneForge.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video → keyframes → overlapping VGGT-Ω windows → one cuNLS-refined sparse map."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import socket
import time
from pathlib import Path
import shutil
from uuid import uuid4

from sceneforge.geometry.inputs import validate_image_paths
from sceneforge.geometry.checkpoint import resolve_checkpoint
from .native_map import native_backend
from .visualization import ReconstructionVisualization
from sceneforge.utils.artifacts import write_json
from sceneforge.video.sampling import VideoFrameSampler
from .windowed import WindowReconstructor, WindowOptions, POLICY

ROOT = Path(__file__).resolve().parents[2]


def artifact_display_path(path: Path) -> Path:
    """Translate paths in Docker's data mount for host-facing log messages only."""
    host_data = os.environ.get('SCENEFORGE_HOST_DATA_DIR')
    if not host_data or not Path(host_data).is_absolute():
        return path
    try:
        relative = path.resolve().relative_to((ROOT / 'data').resolve())
    except ValueError:
        return path  # A custom output outside the mount has no known host path.
    return Path(host_data) / relative


def signature(video: Path, keyframe_config: Path) -> dict:
    digest = hashlib.sha256()
    files = [*Path(__file__).parent.glob("*.py"),
             ROOT / "sceneforge/geometry/vggt_omega.py", ROOT / "sceneforge/geometry/storage.py"]
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    backend = native_backend()
    digest.update(Path(backend.__file__).read_bytes())
    executable = shutil.which("sceneforge-match-pairs")
    if executable:
        digest.update(Path(executable).read_bytes())
    stat = video.stat()
    return {"video": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "implementation_sha256": digest.hexdigest(), "keyframe_config": keyframe_config.read_text()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--resume', type=Path, help='Reuse inputs/VGGT-Ω windows from this pipeline; recompute map in a fresh attempt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--loop-closure', action=argparse.BooleanOptionalAction, default=None,
                        help='Retrieve and verify distant image pairs (default); --no-loop-closure uses temporal pairs only')
    parser.add_argument('--window-size', type=int, help='Keyframes per VGGT-Ω window (default 64)')
    parser.add_argument('--overlap', type=int, help='Shared keyframes between windows (default 32)')
    parser.add_argument('--neighbors', type=int, help='Temporal feature matching neighbors (default 4)')
    parser.add_argument('--device', help='cuda: visible GPUs for VGGT-Ω, first GPU for BA; cuda:N: one GPU')
    parser.add_argument('--lm-iterations', type=int, help='Joint BA iteration budget (1–1000; default 1000)')
    parser.add_argument('--dense-voxel-fraction', type=float, help='Voxel width / median scene depth (default 0.01; larger uses less memory)')
    parser.add_argument('--keyframe-config', type=Path)
    parser.add_argument('--diagnostics', action='store_true', help='Check native CUDA Jacobians (slower; no intermediate map dumps)')
    parser.add_argument('--rerun', action=argparse.BooleanOptionalAction, default=True,
                        help='Save pipeline.rrd (default); --no-rerun disables visualization and recording')
    parser.add_argument('--headless', action='store_true',
                        default=not bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')),
                        help='Save the recording without opening a viewer; automatic when no display is available')
    parser.add_argument('--debug', action='store_true', help='Show exception traceback')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger('dinov3').setLevel(logging.WARNING)
    output = None
    reconstructor = None
    viewer_port = 0
    keep_viewer = False
    try:
        native_backend()
        if args.resume:
            if any(value is not None for value in (args.output, args.checkpoint, args.window_size,
                    args.overlap, args.neighbors, args.device, args.keyframe_config, args.loop_closure)):
                raise ValueError('--resume keeps saved input/window settings; only the BA budget, dense voxel fraction and diagnostics may change')
            candidate = args.resume.expanduser().resolve()
            saved = json.loads((candidate / 'request.json').read_text())
            if saved.get('pipeline') != POLICY:
                raise ValueError('This is an older experimental run. Start with --video; decoded/keyframe caches remain reusable')
            video, config = Path(saved['video']), Path(saved['keyframe_config'])
            if signature(video, config) != saved['signature']:
                raise ValueError('Inputs or implementation changed; start a fresh --video run to avoid stale derived inputs')
            options = WindowOptions(**saved['options'])
            if args.dense_voxel_fraction is not None:
                options = replace(options, dense_voxel_fraction=args.dense_voxel_fraction)
            if args.lm_iterations is not None:
                options = replace(options, lm_iterations=args.lm_iterations)
            checkpoint = Path(saved['checkpoint'])
            if (checkpoint.stat().st_size != saved['checkpoint_bytes'] or
                    checkpoint.stat().st_mtime_ns != saved['checkpoint_mtime_ns']):
                raise ValueError('Checkpoint changed; start a fresh run')
            output = candidate
        else:
            video = args.video.expanduser().resolve()
            if not video.is_file():
                raise FileNotFoundError(f'Video not found: {video}')
            config = (args.keyframe_config or ROOT / 'configs/keyframes_raco.json').expanduser().resolve()
            options = WindowOptions(window_size=args.window_size if args.window_size is not None else 64,
                                    overlap=args.overlap if args.overlap is not None else 32,
                                    neighbors=args.neighbors if args.neighbors is not None else 4,
                                    device=args.device or 'cuda',
                                    loop_closure=args.loop_closure if args.loop_closure is not None else True,
                                    lm_iterations=args.lm_iterations if args.lm_iterations is not None else 1000,
                                    dense_voxel_fraction=args.dense_voxel_fraction if args.dense_voxel_fraction is not None else 0.01)
            output = (args.output or ROOT / 'data/output' / datetime.now(timezone.utc).strftime(
                'reconstruction_%Y%m%d_%H%M%S_%f')).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=False)
            checkpoint = resolve_checkpoint(args.checkpoint)
            write_json(output / 'request.json', {'pipeline': POLICY, 'video': str(video),
                       'keyframe_config': str(config), 'checkpoint': str(checkpoint),
                       'checkpoint_bytes': checkpoint.stat().st_size,
                       'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
                       'signature': signature(video, config), 'options': asdict(options)})
        attempt = output / datetime.now(timezone.utc).strftime('map_%Y%m%d_%H%M%S_%f')
        attempt.mkdir()
        write_json(attempt / 'options.json', {**asdict(options), 'diagnostics': args.diagnostics})
        reconstructor = WindowReconstructor(options, output, config, attempt, args.diagnostics)
        if args.rerun:
            recording = attempt / 'pipeline.rrd'
            viewer_port = reconstructor.native.builder.enable_rerun(str(recording), not args.headless)
            reconstructor.visualization = ReconstructionVisualization(reconstructor.native)
            logging.info('Rerun %s; recording: %s', 'recording only' if args.headless else 'live viewer connected',
                         artifact_display_path(recording))
        logging.info('Video → keyframes → %s → VGGT-Ω windows → common map + BA → dense refinement',
                     'temporal and loop tracks' if options.loop_closure else 'temporal tracks')
        if reconstructor.visualization is not None:
            reconstructor.visualization.event("Decoding / selecting keyframes: waiting for ordered candidate images")
        sampler = VideoFrameSampler(keyframe_config=config)
        folder = output / 'input_frames'
        if (folder / 'manifest.json').is_file():
            sampled = sampler._read_manifest(folder)
        else:
            if folder.exists():
                folder.rename(folder.with_name('input_frames.failed_' + uuid4().hex[:8]))
            sampled = sampler.sample(video, folder, on_progress=reconstructor.visualization.selection
                                     if reconstructor.visualization is not None else None)
        if reconstructor.visualization is not None:
            reconstructor.visualization.keyframes(sampled.paths)
        validate_image_paths(sampled.paths)
        if not (output / 'selection.json').exists():
            write_json(output / 'selection.json', {'frames': len(sampled.paths),
                       'decoded_candidates': sampled.candidate_frame_count,
                       'source_frame_indices': sampled.source_frame_indices,
                       'timestamps_seconds': sampled.timestamps_seconds})
        result = reconstructor.run(list(sampled.paths), checkpoint, sampled.timestamps_seconds)
        logging.info('%s: %d/%d keyframes. Artifacts: %s', result['status'], result['registered_frames'],
                     result['input_frames'], artifact_display_path(output))
        if options.loop_closure:
            loops = result.get('loop_closure', {})
            logging.info('Optimized map retains %d loop landmarks across %d verified pairs',
                         loops.get('retained_loop_landmarks', 0), loops.get('retained_loop_pairs', 0))
        if args.rerun:
            logging.info('Rerun recording: %s', artifact_display_path(recording))
        keep_viewer = bool(viewer_port)
        return 0 if result['status'] == 'complete' else 1
    except (Exception, KeyboardInterrupt) as error:
        logging.error('%s%s', str(error) or 'Interrupted', f'\nArtifacts retained in {artifact_display_path(output)}' if output else '')
        if args.debug:
            logging.exception('Reconstruction traceback')
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if reconstructor is not None and args.rerun:
            # Flush and close the recording before waiting, including on failures.
            reconstructor.native.builder.close_rerun()
        reconstructor = None
        if keep_viewer:
            logging.info('Reconstruction saved. Close the Rerun viewer or press Ctrl+C to exit.')
            try:
                while True:
                    try:
                        with socket.create_connection(('127.0.0.1', viewer_port), timeout=1):
                            pass
                    except OSError:
                        break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass  # The completed reconstruction's exit status is preserved.


if __name__ == '__main__':
    raise SystemExit(main())
