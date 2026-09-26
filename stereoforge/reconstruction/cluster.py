"""Refine one saved VGGT cluster with StereoForge's C++/CUDA cuNLS backend."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from stereoforge.geometry.storage import load_sequence
from stereoforge.refinement.cunls import CuNLSBundleAdjuster, CuNLSOptions
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint, reprojection_error
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.point_cloud import write_ply


class ClusterInitializer:
    def __init__(self, run: Path, section: Path, output: Path, device: int, check_jacobians: bool = False,
                 diagnostics: bool = True) -> None:
        self.run, self.section, self.output = run, section, output
        self.diagnostics = diagnostics
        self.optimizer = CuNLSBundleAdjuster(device, CuNLSOptions(check_jacobians=check_jacobians))
        self.images: dict[int, Path] = {}

    def initialize(self) -> SparseModel:
        sequence = load_sequence(self.section)
        frames = {f.frame_index: f for f in sequence.frames}
        if len(frames) != len(sequence.frames) or len(frames) < 3:
            raise ValueError('Saved cluster must have at least three unique frames')
        cameras = {f.frame_index: SparseCamera(f.camera_to_world.numpy().astype(float),
                   f.intrinsics.numpy().astype(float), f.image_size_hw) for f in sequence.frames}
        previews = self.output / 'images'
        if self.diagnostics:
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
            path = previews / f'{frame.frame_index:06d}.png' if self.diagnostics else original
            if self.diagnostics:
                Image.fromarray(rgb).save(path)
            self.images[frame.frame_index] = path
            colors[frame.frame_index] = rgb
        points = []
        track_ids = []
        candidate_count = 0
        # Tracks are measured image correspondences; initialize positions from
        # VGGT depth, then retain at least three positive-depth observations.
        with (self.run / 'global_tracks.jsonl').open() as stream:
            for track_id, line in enumerate(stream):
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
                track_ids.append(track_id)
        write_json(self.output / 'track_ids.json', {'point_track_ids': track_ids})
        supported = {f for p in points for f in p.observations}
        write_json(self.output / 'initialization.json', {
            'section': str(self.section), 'candidate_tracks': candidate_count, 'landmarks': len(points),
            'unsupported_frames': sorted(set(cameras)-supported), 'pre_ba_max_reprojection_pixels': 14,
            'method': 'Median of VGGT depth-unprojected track observations',
            'backend': 'StereoForge custom cuNLS; not GTSfM or pyCuSFM'})
        if supported != cameras.keys() or len(points) < 60:
            raise ValueError('Cluster initialization lacks all cameras or 60 landmarks; inspect initialization.json')
        return SparseModel(cameras, tuple(points))

    def export_diagnostics(self, model: SparseModel, destination: Path, title: str, report: dict) -> None:
        if not self.diagnostics:
            return
        destination.mkdir()
        ids = sorted(model.cameras)
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
        write_json(destination / 'metadata.json', metadata)
        write_ply(destination / 'point_cloud.ply', xyz, rgb)

    def solve(self) -> tuple[SparseModel, dict]:
        self.optimizer.preflight()
        model = self.initialize()
        model.write(self.output / 'initialized_sparse', sorted(model.cameras))
        self.export_diagnostics(model, self.output / 'before', 'VGGT cluster initialization', {'input_frames': len(model.cameras)})
        refined, report = self.optimizer.optimize(model, self.output / 'ba')
        if refined.points:
            self.export_diagnostics(refined, self.output / 'after', 'cuNLS local BA', report)
        write_json(self.output / 'report.json', report)
        return refined, report
