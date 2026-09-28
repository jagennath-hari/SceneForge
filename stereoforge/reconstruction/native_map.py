"""Tensor loading and final export for the in-process C++ common-map builder."""

import importlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from stereoforge.geometry.storage import load_sequence
from stereoforge.utils.artifacts import write_json
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint


def native_backend():
    try:
        module = importlib.import_module('_stereoforge_map')
    except ImportError as error:
        raise RuntimeError('Rebuild Docker to install the native common-map builder (_stereoforge_map)') from error
    if getattr(module, "api_version", None) != 18:
        raise RuntimeError('Native common-map module is outdated. Stop the old container, rebuild Docker, and start a new container.')
    return module


class WindowRejected(RuntimeError):
    """Native candidate rejection; the accepted map is unchanged."""

    def __init__(self, message: str, unanchored_frames=(), supported_frames=()) -> None:
        super().__init__(message)
        self.unanchored_frames = frozenset(unanchored_frames)
        self.supported_frames = tuple(supported_frames)


class NativeMap:
    def __init__(self, device: int, iterations: int, diagnostics: bool) -> None:
        self.backend = native_backend()
        self.builder = self.backend.MapBuilder(device, iterations, diagnostics)
        self.stage_seconds: dict[str, float] = {}
        self.last_stage: str | None = None
        self.alignment_warnings: list[str] = []
        self._frame_cache: OrderedDict[Path, dict] = OrderedDict()

    def load_tracks(self, path: Path) -> None:
        with path.open() as stream:
            tracks = [{int(frame): np.asarray(uv, dtype=np.float64)
                       for frame, uv in json.loads(line).items()} for line in stream]
        self.builder.set_tracks(tracks)

    def _load_frames(self, path: Path, images: list[Path]) -> dict:
        if path in self._frame_cache:
            self._frame_cache.move_to_end(path)
            return self._frame_cache[path]
        sequence = load_sequence(path)
        if sequence.processed_rgb is None:
            raise ValueError('Window has no processed RGB for feature-grid verification')
        frames = {}
        for index, frame in enumerate(sequence.frames):
            rgb = np.ascontiguousarray((sequence.processed_rgb[index].numpy().transpose(1, 2, 0)*255).round(), dtype=np.uint8)
            with Image.open(images[frame.frame_index]) as image:
                if not np.array_equal(np.asarray(image.convert('RGB')), rgb):
                    raise ValueError(f'Feature/VGGT image grid mismatch at frame {frame.frame_index}')
            frames[frame.frame_index] = self.backend.DepthFrame(frame.frame_index,
                frame.camera_to_world.numpy().astype(np.float64), frame.intrinsics.numpy().astype(np.float64),
                np.ascontiguousarray(frame.depth.numpy(), dtype=np.float32), rgb)
        self._frame_cache[path] = frames
        while len(self._frame_cache) > 2:
            self._frame_cache.popitem(last=False)
        return frames

    def add_window(self, path: Path, images: list[Path], references: dict[int, Path], progress,
                   frame_ids: tuple[int, ...] | None = None) -> None:
        frames = self._load_frames(path, images)
        if frame_ids is not None:
            # Keep the saved prediction's coordinate system. Only the selected
            # cameras enter initialization/BA; source ownership remains per frame.
            frames = {frame: frames[frame] for frame in frame_ids}
        reference_frames = []
        for source in sorted(set(references.values())):
            saved = self._load_frames(source, images)
            reference_frames.extend(saved[frame] for frame, owner in references.items() if owner == source)
        try:
            self._invoke(lambda update: self.builder.add_window(list(frames.values()), reference_frames, update), progress)
        except RuntimeError as error:
            raise WindowRejected(str(error), self.builder.unanchored_frames,
                                 self.builder.supported_initialization_frames) from error

    def rank_windows(self, windows: list[list[int]]) -> list[int]:
        return self.builder.rank_windows(windows)

    def finalize(self, progress, directory: Path, loop_evidence: Path | None = None) -> None:
        """Persist exact solver arrays and fixed targets before releasing pose priors."""
        directory.mkdir(parents=True, exist_ok=True)
        loop_mask: np.ndarray | None = None
        observation_codes: np.ndarray | None = None
        identity_stride = 0
        manifest = {'format_version': 1, 'schedule': [1.0, 0.1, 0.01, 0.0],
                    'status': 'running', 'completed_stages': [],
                    'coordinate_convention': 'normalized world_to_camera; origin/scale in input.json',
                    'stage_observations': 'Every stage uses input.npz observation indices and immutable prior targets'}
        write_json(directory / 'manifest.json', manifest)

        def checkpoint(name: str, document: dict) -> None:
            nonlocal loop_mask, observation_codes, identity_stride
            metadata = document.pop('metadata')
            if name == 'input':
                # Full native IDs are retained, not inferred from PLY point order.
                model = self.builder.map
                camera_ids = document['camera_ids'].tolist()
                track_ids = document['track_ids'].tolist()
                cameras, landmarks = model.cameras, model.landmarks
                original_poses = np.repeat(np.eye(4)[None], len(camera_ids), axis=0)
                for i, identifier in enumerate(camera_ids):
                    original_poses[i, :3, :3] = cameras[identifier].rotation
                    original_poses[i, :3, 3] = cameras[identifier].center
                document['original_camera_to_world'] = original_poses
                document['original_intrinsics'] = np.asarray([cameras[i].intrinsics for i in camera_ids])
                document['image_sizes_hw'] = np.asarray([(cameras[i].height, cameras[i].width) for i in camera_ids], dtype=np.int32)
                document['original_points'] = np.asarray([landmarks[i].position for i in track_ids])
                document['colors'] = np.asarray([landmarks[i].color for i in track_ids], dtype=np.uint8)
                obs_tracks = document['track_ids'][document['observation_point']]
                obs_frames = document['camera_ids'][document['observation_camera']]
                identity_stride = int(max(camera_ids)) + 1
                observation_codes = obs_tracks*identity_stride+obs_frames
                loop_mask = np.zeros(len(obs_tracks), dtype=bool)
                if loop_evidence is not None and loop_evidence.exists():
                    evidence = json.loads(loop_evidence.read_text())
                    support: dict[int, set[int]] = {}
                    for track, frame in zip(obs_tracks.tolist(), obs_frames.tolist()):
                        support.setdefault(track, set()).add(frame)
                    stride = identity_stride
                    endpoints = set()
                    for identifier, pairs in evidence.items():
                        track = int(identifier)
                        seen = support.get(track, set())
                        for a, b in pairs:
                            if a in seen and b in seen:
                                endpoints.add(track*stride+a)
                                endpoints.add(track*stride+b)
                    loop_mask = np.isin(obs_tracks*stride+obs_frames, np.fromiter(sorted(endpoints), dtype=np.int64))
                document['loop_observation_mask'] = loop_mask
                metadata['loop_observations'] = int(loop_mask.sum())
                metadata['loop_mask_definition'] = 'Unique BA observations at endpoints of verified loop pairs retained on the same native landmark'
            else:
                metadata['reprojection_before'] = self._error_summary(document['squared_errors_before'])
                metadata['reprojection_after'] = self._error_summary(document['squared_errors_after'])
                if loop_mask is not None:
                    metadata['loop_reprojection_before'] = self._error_summary(document['squared_errors_before'][loop_mask])
                    metadata['loop_reprojection_after'] = self._error_summary(document['squared_errors_after'][loop_mask])
            temporary = directory / f'{name}.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez(stream, **document)
            temporary.replace(directory / f'{name}.npz')
            write_json(directory / f'{name}.json', metadata)
            if name != 'input':
                manifest['completed_stages'].append(name)
                write_json(directory / 'manifest.json', manifest)

        self.builder.set_ba_checkpoint(checkpoint)
        try:
            self._invoke(self.builder.finalize, progress)
            # Stage arrays are unfiltered solver outputs. Persist the exact
            # membership that survived the native positive-depth/support pass.
            landmarks = self.builder.map.landmarks
            retained_codes = np.fromiter((track*identity_stride+frame
                for track, point in landmarks.items() for frame in point.observations), dtype=np.int64)
            with (directory / 'accepted.npz.tmp').open('wb') as stream:
                np.savez(stream, retained_track_ids=np.asarray(list(landmarks), dtype=np.int64),
                         retained_observation_mask=np.isin(observation_codes, retained_codes))
            (directory / 'accepted.npz.tmp').replace(directory / 'accepted.npz')
            manifest['status'] = 'complete'
            manifest['accepted_state'] = 'stage_4.npz with accepted.npz membership and input.npz identities/normalization'
        except BaseException as error:
            manifest.update(status='failed', error=str(error) or type(error).__name__)
            raise
        finally:
            self.builder.set_ba_checkpoint(None)
            write_json(directory / 'manifest.json', manifest)

    @staticmethod
    def _error_summary(squared: np.ndarray) -> dict:
        errors = np.sqrt(np.asarray(squared, dtype=np.float64))
        finite = errors[np.isfinite(errors)]
        return {'observations': len(errors), 'invalid': int(len(errors)-len(finite)),
                'median_pixels': float(np.median(finite)) if finite.size else None,
                'p90_pixels': float(np.percentile(finite, 90)) if finite.size else None,
                'within_3px': int(np.count_nonzero(errors <= 3)),
                'within_5px': int(np.count_nonzero(errors <= 5))}

    def _invoke(self, operation, progress) -> None:
        stage = None
        started = perf_counter()

        def update(name: str) -> None:
            nonlocal stage, started
            if name.startswith('WARNING: '):
                message = name.removeprefix('WARNING: ')
                if message not in self.alignment_warnings:
                    self.alignment_warnings.append(message)
                    logging.getLogger(__name__).warning('%s', message)
                return
            now = perf_counter()
            if stage is not None:
                self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + now - started
            stage, started = name, now
            self.last_stage = name
            progress(name)

        try:
            operation(update)
        finally:
            if stage is not None:
                self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + perf_counter() - started

    @property
    def accepted_windows(self) -> int:
        return self.builder.accepted_windows

    def export(self) -> SparseModel | None:
        if not self.accepted_windows:
            return None
        model = self.builder.map
        cameras = {}
        for frame, camera in model.cameras.items():
            pose = np.eye(4)
            pose[:3, :3] = camera.rotation
            pose[:3, 3] = camera.center
            cameras[frame] = SparseCamera(pose, camera.intrinsics, (camera.height, camera.width))
        points = tuple(SparsePoint(point.position, np.asarray(point.color, dtype=np.uint8), point.observations)
                       for point in model.landmarks.values())
        return SparseModel(cameras, points)
