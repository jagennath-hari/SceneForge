"""Video → keyframes → overlapping VGGT windows → one cuNLS-refined sparse map."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
from uuid import uuid4

from stereoforge.geometry.inputs import validate_image_paths
from stereoforge.geometry.checkpoint import resolve_checkpoint
from .native_map import native_backend
from .visualization import ReconstructionVisualization
from stereoforge.utils.artifacts import write_json
from stereoforge.video.sampling import VideoFrameSampler
from .windowed import WindowReconstructor, WindowOptions, POLICY

ROOT = Path(__file__).resolve().parents[2]


def signature(video: Path, keyframe_config: Path) -> dict:
    digest = hashlib.sha256()
    files = [*Path(__file__).parent.glob("*.py"), *(ROOT / "stereoforge/refinement").glob("*.py"),
             ROOT / "stereoforge/geometry/vggt_omega.py", ROOT / "stereoforge/geometry/storage.py"]
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    backend = native_backend()
    digest.update(Path(backend.__file__).read_bytes())
    executable = shutil.which("stereoforge-match-pairs")
    if executable:
        digest.update(Path(executable).read_bytes())
    stat = video.stat()
    return {"video": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "implementation_sha256": digest.hexdigest(), "keyframe_config": keyframe_config.read_text()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--resume', type=Path, help='Reuse inputs/VGGT windows from this pipeline; recompute map in a fresh attempt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--window-size', type=int, help='Keyframes per VGGT window (default 32)')
    parser.add_argument('--overlap', type=int, help='Shared keyframes between windows (default 16)')
    parser.add_argument('--neighbors', type=int, help='Temporal feature matching neighbors (default 4)')
    parser.add_argument('--device', help='cuda: visible GPUs for VGGT, first GPU for BA; cuda:N: one GPU')
    parser.add_argument('--lm-iterations', type=int, help='Joint BA iteration budget (default 500)')
    parser.add_argument('--dense-voxel-fraction', type=float, help='Voxel width / median scene depth (default 0.01; larger uses less memory)')
    parser.add_argument('--keyframe-config', type=Path)
    parser.add_argument('--diagnostics', action='store_true', help='Check native CUDA Jacobians (slower; no intermediate map dumps)')
    parser.add_argument('--rerun', action=argparse.BooleanOptionalAction, default=True,
                        help='Stream to Rerun and save pipeline.rrd (default); --no-rerun runs headless')
    parser.add_argument('--debug', action='store_true', help='Show exception traceback')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger('dinov3').setLevel(logging.WARNING)
    output = None
    try:
        native_backend()
        if args.resume:
            if any(value is not None for value in (args.output, args.checkpoint, args.window_size,
                    args.overlap, args.neighbors, args.device, args.keyframe_config)):
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
            options = WindowOptions(window_size=args.window_size if args.window_size is not None else 32,
                                    overlap=args.overlap if args.overlap is not None else 16,
                                    neighbors=args.neighbors if args.neighbors is not None else 4,
                                    device=args.device or 'cuda',
                                    lm_iterations=args.lm_iterations if args.lm_iterations is not None else 500,
                                    dense_voxel_fraction=args.dense_voxel_fraction if args.dense_voxel_fraction is not None else 0.01)
            output = (args.output or ROOT / 'data/intermediate' / datetime.now(timezone.utc).strftime(
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
            reconstructor.native.builder.enable_rerun(str(recording))
            reconstructor.visualization = ReconstructionVisualization(reconstructor.native)
            logging.info('Rerun live viewer connected; recording also saved to %s', recording)
        logging.info('Video → keyframes → VGGT windows → graph-ordered map → shared-calibration BA → dense refinement')
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
                     result['input_frames'], output)
        if args.rerun:
            logging.info('Rerun recording: %s', recording)
        return 0 if result['status'] == 'complete' else 1
    except (Exception, KeyboardInterrupt) as error:
        logging.error('%s%s', str(error) or 'Interrupted', f'\nArtifacts retained in {output}' if output else '')
        if args.debug:
            logging.exception('Reconstruction traceback')
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
