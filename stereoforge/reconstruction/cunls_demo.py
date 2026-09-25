"""Refine one saved VGGT cluster with StereoForge's C++/CUDA cuNLS backend."""

import argparse
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from stereoforge.geometry.adaptive import _load
from stereoforge.refinement.cunls import CuNLSBundleAdjuster, CuNLSOptions
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint, reprojection_error
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.visualization import GeometryReportWriter


class ClusterDiagnostic:
    def __init__(self, run: Path, section: Path, output: Path, device: int, check_jacobians: bool = False) -> None:
        self.run, self.section, self.output = run, section, output
        self.optimizer = CuNLSBundleAdjuster(device, CuNLSOptions(check_jacobians=check_jacobians))
        self.images: dict[int, Path] = {}

    def initialize(self) -> SparseModel:
        sequence = _load(self.section)
        frames = {f.frame_index: f for f in sequence.frames}
        if len(frames) != len(sequence.frames) or len(frames) < 3:
            raise ValueError('Saved cluster must have at least three unique frames')
        cameras = {f.frame_index: SparseCamera(f.camera_to_world.numpy().astype(float),
                   f.intrinsics.numpy().astype(float), f.image_size_hw) for f in sequence.frames}
        previews = self.output / 'images'
        previews.mkdir()
        colors = {}
        for i, frame in enumerate(sequence.frames):
            if sequence.processed_rgb is None:
                raise ValueError('Saved cluster lacks processed RGB; cannot verify feature coordinate grid')
            rgb = (sequence.processed_rgb[i].numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
            original = self.run / 'processed' / f'{frame.frame_index:06d}.png'
            with Image.open(original) as image:
                if not np.array_equal(np.array(image.convert('RGB')), rgb):
                    raise ValueError(f'Feature/VGGT processed image mismatch at frame {frame.frame_index}')
            path = previews / f'{frame.frame_index:06d}.png'
            Image.fromarray(rgb).save(path)
            self.images[frame.frame_index] = path
            colors[frame.frame_index] = rgb
        points = []
        candidate_count = 0
        # Tracks are measured image correspondences; initialize positions from
        # VGGT depth, then retain at least three positive-depth observations.
        with (self.run / 'global_tracks.jsonl').open() as stream:
            for line in stream:
                track = {int(f): np.asarray(uv, dtype=float) for f, uv in json.loads(line).items() if int(f) in frames}
                if len(track) < 3:
                    continue
                candidate_count += 1
                positions = []
                for f, uv in track.items():
                    frame = frames[f]
                    h, w = frame.image_size_hw
                    if uv.shape != (2,) or not np.isfinite(uv).all() or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                        continue
                    x, y = np.rint(uv).astype(int)
                    x, y = min(x, w-1), min(y, h-1)
                    depth = float(frame.depth[y, x])
                    if not np.isfinite(depth) or depth <= 0:
                        continue
                    camera = cameras[f]
                    ray = np.linalg.solve(camera.intrinsics, [*uv, 1])
                    positions.append(camera.pose[:3, :3] @ (ray*depth) + camera.pose[:3, 3])
                if not positions:
                    continue
                xyz = np.median(positions, axis=0)
                valid = {f: uv for f, uv in track.items() if reprojection_error(cameras[f], xyz, uv) <= 14}
                if len(valid) < 3:
                    continue
                f, uv = next(iter(valid.items()))
                h, w = colors[f].shape[:2]
                x, y = np.clip(np.rint(uv).astype(int), [0, 0], [w-1, h-1])
                points.append(SparsePoint(xyz, colors[f][y, x].copy(), valid))
        supported = {f for p in points for f in p.observations}
        write_json(self.output / 'initialization.json', {
            'section': str(self.section), 'candidate_tracks': candidate_count, 'landmarks': len(points),
            'unsupported_frames': sorted(set(cameras)-supported), 'pre_ba_max_reprojection_pixels': 14,
            'method': 'Median of VGGT depth-unprojected track observations',
            'backend': 'StereoForge custom cuNLS; not GTSfM or pyCuSFM'})
        if supported != cameras.keys() or len(points) < 60:
            raise ValueError('Cluster initialization lacks all cameras or 60 landmarks; inspect initialization.json')
        return SparseModel(cameras, tuple(points))

    def viewer(self, model: SparseModel, destination: Path, title: str, report: dict) -> None:
        destination.mkdir()
        ids = sorted(model.cameras)
        order = {f: i for i, f in enumerate(ids)}
        frames = [{'frame_index': f, 'source': f'Keyframe {f}', 'processed_size_hw': c.size_hw,
                   'camera_to_world': c.pose.tolist(), 'intrinsics': c.intrinsics.tolist(),
                   'valid_fraction': 0, 'depth_p50': None,
                   'previews': {'rgb': f'../images/{f:06d}.png'}} for f, c in sorted(model.cameras.items())]
        metadata = {'format_version': 1, 'units': 'reconstruction_units', 'meters_per_unit': None,
                    'reconstruction_name': title, 'sparse_only': True, 'frames': frames,
                    'input_frames': report['input_frames'], 'sparse_statistics': report,
                    'dense_depth_refined': False, 'provenance': {}}
        indices = np.linspace(0, len(model.points)-1, min(240000, len(model.points)), dtype=int)
        points = [model.points[i] for i in indices]
        xyz = np.asarray([p.xyz for p in points], dtype=float).reshape(-1, 3)
        rgb = np.asarray([p.rgb for p in points], dtype=np.uint8).reshape(-1, 3)
        appeared = np.asarray([min(order[f] for f in p.observations) for p in points], dtype=np.int64)
        write_json(destination / 'metadata.json', metadata)
        GeometryReportWriter._write_ply(destination / 'point_cloud.ply', xyz, rgb)
        GeometryReportWriter._write_viewer(destination, metadata, xyz, rgb, appeared)

    def execute(self) -> dict:
        self.optimizer.preflight()
        model = self.initialize()
        model.write(self.output / 'initialized_sparse', sorted(model.cameras))
        self.viewer(model, self.output / 'before', 'VGGT cluster initialization', {'input_frames': len(model.cameras)})
        refined, report = self.optimizer.optimize(model, self.output / 'ba')
        if refined.points:
            self.viewer(refined, self.output / 'after', 'cuNLS local BA', report)
        write_json(self.output / 'report.json', report)
        page = ('<!doctype html><meta charset="utf-8"><title>cuNLS cluster diagnostic</title>'
                '<h1>cuNLS cluster diagnostic</h1><p>This is a local sparse cluster, not a full reconstruction.</p>'
                '<p><a href="before/index.html">Before BA</a> · '
                + ('<a href="after/index.html">After BA</a> · ' if refined.points else '')
                + '<a href="ba/solver.log">Solver log</a> · <a href="report.json">Full diagnostics</a></p>'
                + '<pre>' + html.escape(json.dumps(report, indent=2)) + '</pre>')
        (self.output / 'index.html').write_text(page)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing run with global_tracks.jsonl and processed/')
    parser.add_argument('--section', type=Path, help='Saved VGGT .pt cluster; default: RUN/seed_vggt/0.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', type=int, default=0, help='Logical index within CUDA_VISIBLE_DEVICES')
    parser.add_argument('--check-jacobians', action='store_true', help='Check native pixel Jacobians before optimization')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    output = None
    try:
        CuNLSBundleAdjuster.preflight()
        run = args.run.expanduser().resolve()
        section = args.section.expanduser().resolve() if args.section else run / 'seed_vggt/0.pt'
        if not section.is_file() or not (run / 'global_tracks.jsonl').is_file():
            raise ValueError('Saved VGGT cluster or global tracks file is missing')
        candidate = (args.output or run / datetime.now(timezone.utc).strftime('cunls_%Y%m%d_%H%M%S_%f')).resolve()
        candidate.mkdir(parents=True, exist_ok=False)
        output = candidate
        report = ClusterDiagnostic(run, section, output, args.device, args.check_jacobians).execute()
        logging.info('%s: %d/%d cluster cameras. Open %s', report['status'], report['registered_frames'],
                     report['input_frames'], output / 'index.html')
        return 0 if report['status'] == 'diagnostic_complete' else 1
    except (Exception, KeyboardInterrupt) as error:
        logging.error('%s%s', str(error) or 'Interrupted', f'\nArtifacts retained in {output}' if output else '')
        if args.debug:
            logging.exception('cuNLS diagnostic traceback')
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
