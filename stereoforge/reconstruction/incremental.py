"""VGGT seed followed by registration into one expanding sparse map."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from stereoforge.geometry.adaptive import _load
from stereoforge.refinement.bundle_adjustment import model_statistics, require_connected
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group
from .frontend import ReconstructionFrontend
from .inference import infer_clusters
from .map_registration import MapRegistration, key
from .triangulation import TrackTriangulator
from .view_graph import Cluster, VerifiedGraph

POLICY = 'incremental_map_v1'


class IncrementalMapper(MapRegistration):
    def __init__(self, frontend: ReconstructionFrontend, tracks: list[dict], model: SparseModel,
                 count: int, calibration: np.ndarray, size: tuple[int, int], state: dict | None) -> None:
        self.run = self.output = frontend.output
        self.optimizer, self.package = frontend.optimizer, frontend.package
        self.model = model
        self.ids = list(range(count))
        self.tracks, self.track_ids = tracks, list(range(len(tracks)))
        self.lookup = {}
        for t, track in enumerate(tracks):
            for frame, uv in track.items():
                self.lookup.setdefault(key(frame, uv), set()).add(t)
        self.calibration = {f: SparseCamera(np.eye(4), calibration.copy(), size) for f in self.ids}
        self.intrinsics, self.size = calibration, size
        self.holdouts = {} if state is None else {int(f): ts for f, ts in state['holdouts'].items()}
        self.excluded = {(t, f) for f, ts in self.holdouts.items() for t in ts}
        self.targets = sorted(set(self.ids) - set(model.cameras)) if state is None else state['targets']
        self.report = ({'status': 'running', 'pipeline': POLICY, 'input_frames': count,
                        'anchor_frames': sorted(model.cameras), 'target_frames': self.targets,
                        'attempts': [], 'ba_rounds': [], 'units': 'reconstruction_units',
                        'calibration_policy': 'Seed median PINHOLE intrinsics initialize later frames; fixed processed image size, no zoom model',
                        'validation_scope': 'Held-out target pixels excluded from registration, triangulation and BA; seed has no independent ground truth'}
                       if state is None else state['report'])
        require_connected(model)

    def checkpoint(self, directory: Path, tracks_digest: str) -> None:
        state = {'policy': POLICY, 'directory': str(directory.relative_to(self.output)),
                 'tracks_sha256': tracks_digest, 'input_frames': len(self.ids),
                 'intrinsics': self.intrinsics.tolist(), 'size_hw': self.size,
                 'holdouts': self.holdouts, 'targets': self.targets, 'report': self.report}
        temporary = self.output / 'incremental_state.tmp'
        write_json(temporary, state)
        temporary.replace(self.output / 'incremental_state.json')

    def optimize(self, tracks_digest: str) -> None:
        seed = self.model
        directory = self.output / 'map_ba' / uuid4().hex
        optimized = self.optimizer.optimize_sparse(seed, directory, self.ids, self.package)
        self.model = optimized
        validation = self.validate()
        lost = sorted(seed.cameras.keys() - optimized.cameras.keys())
        record = {'directory': str(directory.relative_to(self.output)), 'registered_frames': len(optimized.cameras),
                  'lost_cameras': lost, 'validation': validation}
        write_json(directory / 'map_validation.json', record)
        if lost or not validation['passed']:
            # The last committed map remains resumable; the rejected BA is retained separately.
            self.model = seed
            self.report.update(status='failed', rejected_ba=record)
            self.save_report()
            raise ValueError(f'Map BA lost cameras or failed held-out checks; inspect {directory / "map_validation.json"}')
        self.report['ba_rounds'].append(record)
        self.report['status'] = 'running'
        self.save_report()
        self.checkpoint(directory / 'workspace/sparse', tracks_digest)

    def grow(self, tracks_digest: str) -> dict:
        pass_id = 0
        with progress_group('Incremental reconstruction'):
            with Progress('Registered cameras', len(self.ids), 'frame') as progress:
                progress.advance(len(self.model.cameras))
                while len(self.model.cameras) < len(self.ids):
                    missing = set(self.ids) - self.model.cameras.keys()
                    registered = sorted(self.model.cameras)
                    # Grow from the current map in either temporal direction.
                    candidates = sorted(missing, key=lambda f: (min(abs(f-r) for r in registered), f))
                    accepted = pending = 0
                    for frame in candidates:
                        progress.status(f'{len(self.model.cameras)}/{len(self.ids)} registered; trying frame {frame}')
                        if self.register(frame, pass_id):
                            accepted += 1
                            pending += 1
                            self.extend_tracks()
                            progress.advance()
                            if pending == 8:
                                self.optimize(tracks_digest)
                                pending = 0
                    if pending:
                        self.optimize(tracks_digest)
                    if not accepted:
                        break
                    pass_id += 1
        missing = sorted(set(self.ids) - self.model.cameras.keys())
        self.report.update(status='partial' if missing else 'complete', missing_frames=missing,
                           registered_frames=len(self.model.cameras), dense_depth_refined=False,
                           reconstruction_name='Incremental pyCuSFM', statistics=model_statistics(self.model))
        self.save_report()
        return self.report


class IncrementalReconstructor(ReconstructionFrontend):
    def run(self, paths: list[Path], checkpoint: Path, timestamps: tuple | None = None) -> dict:
        if len(paths) < 6:
            raise ValueError('At least six selected keyframes are required')
        self.timestamps = timestamps
        self.images = self._prepare_images(paths)
        track_path = self.output / 'global_tracks.jsonl'
        if not track_path.exists():
            files = self._match(len(paths))
            graph = VerifiedGraph(len(paths))
            with Progress('Building verified image graph and tracks'):
                graph.read(files)
            write_json(self.output / 'graph.json', graph.summary)
            temporary = track_path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                for track in graph.tracks:
                    stream.write(json.dumps({str(f): uv.tolist() for f, uv in track.items()}) + '\n')
            temporary.replace(track_path)
            tracks = graph.tracks
        else:
            with track_path.open() as stream:
                tracks = [{int(f): np.asarray(uv, dtype=float) for f, uv in json.loads(line).items()} for line in stream]
        digest = hashlib.sha256(track_path.read_bytes()).hexdigest()
        (self.output / 'map_ba').mkdir(exist_ok=True)
        state_path = self.output / 'incremental_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state is not None:
            if (state['policy'] != POLICY or state['tracks_sha256'] != digest
                    or state['input_frames'] != len(paths)):
                raise ValueError('Saved incremental state does not match the current inputs/tracks/policy')
            model = SparseModel.read(self.output / state['directory'], list(range(len(paths))))
            intrinsics, size = np.asarray(state['intrinsics']), tuple(state['size_hw'])
        else:
            ids = list(range(min(self.options.cluster_size, len(paths))))
            with Progress('Preparing VGGT seed') as progress:
                progress.status(f'{len(ids)} initial keyframes; later poses use map registration')
            infer_clusters(checkpoint, paths, [Cluster(0, ids)], self.output / 'seed_vggt', [self.devices[0]])
            sequence = _load(self.output / 'seed_vggt/0.pt')
            if [f.frame_index for f in sequence.frames] != ids:
                raise ValueError('Saved VGGT seed frame identities differ')
            cameras = {}
            for i, frame in enumerate(sequence.frames):
                with Image.open(self.images[frame.frame_index]) as image:
                    pixels = np.array(image)
                expected = (sequence.processed_rgb[i].numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
                if not np.array_equal(pixels, expected):
                    raise ValueError('VGGT seed and feature tracks use different image grids')
                cameras[frame.frame_index] = SparseCamera(frame.camera_to_world.numpy().astype(float),
                                                        frame.intrinsics.numpy().astype(float), frame.image_size_hw)
            del sequence
            seed, details = TrackTriangulator(tracks, self.images).build(cameras, max_error=3)
            write_json(self.output / 'seed_triangulation.json', details)
            if len(seed.cameras) != len(ids) or len(seed.points) < 60:
                raise ValueError('Seed lacks connected support for every seed camera; inspect seed_triangulation.json')
            directory = self.output / 'map_ba' / ('seed_' + uuid4().hex)
            model = self.optimizer.optimize_sparse(seed, directory, list(range(len(paths))), self.package)
            if set(model.cameras) != set(ids) or len(model.points) < 60:
                raise ValueError('Seed BA lost support; choose a better initial video segment')
            intrinsics = np.median(np.array([c.intrinsics for c in model.cameras.values()]), axis=0)
            size = next(iter(model.cameras.values())).size_hw
        mapper = IncrementalMapper(self, tracks, model, len(paths), intrinsics, size, state)
        if state is None:
            mapper.save_report()
            mapper.checkpoint(directory / 'workspace/sparse', digest)
        result = mapper.grow(digest)
        # Every published camera belongs to the validated expanding map. No other
        # section geometry is transformed or silently inserted into the viewer.
        self._viewer(mapper.model, result, timestamps)
        return result
