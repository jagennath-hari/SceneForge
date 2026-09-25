"""Two overlapping VGGT clusters, local cuNLS BA, validated alignment and joint BA."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import logging
from pathlib import Path
import shutil

from stereoforge.geometry.adaptive import _load
from stereoforge.refinement.cunls import CuNLSBundleAdjuster, CuNLSOptions
from stereoforge.refinement.sparse_model import SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from stereoforge.video.sampling import VideoFrameSampler
from .cunls_demo import ClusterDiagnostic
from .inference import infer_clusters
from .merge_validation import MergeValidation
from .registration import align_sparse_sections, merge_sparse_sections, shared_landmarks
from .view_graph import Cluster

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RefinedCluster:
    model: SparseModel
    track_ids: tuple[int, ...]
    diagnostic: ClusterDiagnostic


class PairDiagnostic:
    """Keep every stage on disk; never publish an unvalidated merge as complete."""

    def __init__(self, run: Path, output: Path, device: int, check_jacobians: bool) -> None:
        self.run, self.output, self.device = run, output, device
        self.check_jacobians = check_jacobians
        self.report: dict = {
            'status': 'running', 'stage': 'preparing', 'input_run': str(run),
            'scope': 'Two-cluster sparse diagnostic; not a full-sequence or dense reconstruction',
            'backend': 'Custom C++/CUDA cuNLS local and joint BA; CPU Sim(3) registration',
        }

    def publish(self) -> None:
        write_json(self.output / 'report.json', self.report)
        links = [('reference/before/index.html', 'Reference before local BA'),
                 ('reference/after/index.html', 'Reference after local BA'),
                 ('local/before/index.html', 'Local before local BA'),
                 ('local/after/index.html', 'Local after local BA'),
                 ('aligned/index.html', 'Aligned local cluster'),
                 ('before_joint/index.html', 'Combined before joint BA'),
                 ('after_joint/index.html', 'Combined after joint BA'),
                 ('alignment.json', 'Alignment checks'), ('validation.json', 'Joint BA validation'),
                 ('reference/ba/solver.log', 'Reference solver log'),
                 ('local/ba/solver.log', 'Local solver log'), ('joint_ba/solver.log', 'Joint solver log')]
        items = ''.join(f'<li><a href="{path}">{label}</a></li>' for path, label in links
                        if (self.output / path).is_file())
        page = ('<!doctype html><meta charset="utf-8"><title>cuNLS pair diagnostic</title>'
                '<h1>cuNLS two-cluster diagnostic</h1><p>' + html.escape(self.report['scope'])
                + '</p><ul>' + items + '</ul><pre>' + html.escape(json.dumps(self.report, indent=2)) + '</pre>')
        (self.output / 'index.html').write_text(page, encoding='utf-8')

    def prepare(self, reference: Path | None, local: Path | None, start: int,
                cluster_size: int, overlap: int, checkpoint: Path | None) -> tuple[Path, Path]:
        if not (self.run / 'global_tracks.jsonl').is_file():
            raise ValueError('Run must contain global_tracks.jsonl and processed images')
        with (self.run / 'global_tracks.jsonl').open('rb') as stream:
            self.report['tracks_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if reference is not None and local is not None:
            return reference.expanduser().resolve(), local.expanduser().resolve()
        if start < 0 or cluster_size < 8 or not 6 <= overlap < cluster_size:
            raise ValueError('Require start >= 0, cluster-size >= 8 and 6 <= overlap < cluster-size')
        directory = self.run / 'input_frames'
        manifest = json.loads((directory / 'manifest.json').read_text())
        frames = VideoFrameSampler(max_edge=manifest['max_edge'])._read_manifest(directory)
        end = start + 2*cluster_size - overlap
        if end > len(frames.paths):
            raise ValueError(f'Pair needs keyframes [{start}, {end}); only {len(frames.paths)} available')
        clusters = [Cluster(0, list(range(start, start+cluster_size))),
                    Cluster(1, list(range(start+cluster_size-overlap, end)))]
        destination = self.output / 'vggt'
        destination.mkdir()
        seed = self.run / 'seed_vggt/0.pt'
        if seed.is_file():
            sequence = _load(seed)
            if [frame.frame_index for frame in sequence.frames] == clusters[0].frames:
                shutil.copyfile(seed, destination / '0.pt')
            del sequence
        saved = json.loads((self.run / 'request.json').read_text())
        weights = checkpoint.expanduser().resolve() if checkpoint else Path(saved['checkpoint'])
        if not weights.is_file():
            raise ValueError('Saved checkpoint is unavailable; supply --checkpoint with an authorized local checkpoint')
        if checkpoint is None and (weights.stat().st_size != saved['checkpoint_bytes']
                                  or weights.stat().st_mtime_ns != saved['checkpoint_mtime_ns']):
            raise ValueError('Saved checkpoint changed; use --checkpoint explicitly or restore the original')
        self.report['checkpoint'] = str(weights)
        self.report['requested_frame_ids'] = [cluster.frames for cluster in clusters]
        self.publish()
        infer_clusters(weights, list(frames.paths), clusters, destination, [f'cuda:{self.device}'])
        return destination / '0.pt', destination / '1.pt'

    def refine(self, name: str, section: Path) -> RefinedCluster:
        self.report['stage'] = f'{name}_local_ba'
        self.publish()
        directory = self.output / name
        directory.mkdir()
        diagnostic = ClusterDiagnostic(self.run, section, directory, self.device, self.check_jacobians)
        model, report = diagnostic.solve()
        self.report[name] = report
        if report['status'] != 'diagnostic_complete':
            raise ValueError(f'{name} local BA is partial; inspect {directory / "report.json"}')
        initial_ids = json.loads((directory / 'track_ids.json').read_text())['point_track_ids']
        identities = [initial_ids[i] for i in report['retained_point_indices']]
        if len(identities) != len(model.points) or len(set(identities)) != len(identities):
            raise ValueError('Local BA changed retained landmark identities')
        write_json(directory / 'optimized_track_ids.json', {'point_track_ids': identities})
        return RefinedCluster(model, tuple(identities), diagnostic)

    def execute(self, sections: tuple[Path, Path]) -> dict:
        frame_sets = []
        for section in sections:
            sequence = _load(section)
            ids = [frame.frame_index for frame in sequence.frames]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate global frame IDs in saved cluster')
            frame_sets.append(set(ids))
            del sequence
        shared = frame_sets[0] & frame_sets[1]
        if len(shared) < 6 or not frame_sets[0]-shared or not frame_sets[1]-shared:
            raise ValueError('Need six shared cameras and distinct non-overlap frames in both clusters')
        self.report['input_frames'] = len(frame_sets[0] | frame_sets[1])
        self.report['shared_frames'] = sorted(shared)
        self.report['sections'] = list(map(str, sections))
        reference = self.refine('reference', sections[0])
        local = self.refine('local', sections[1])
        pairs = shared_landmarks(reference.model, local.model)
        if any(reference.track_ids[i] != local.track_ids[j] for i, j in pairs):
            raise ValueError('Shared image observations disagree with global track identities')
        self.report['stage'] = 'alignment'
        self.publish()
        alignment: dict = {'status': 'running', 'association': 'Shared measured pixels, verified against global track IDs'}
        try:
            with Progress('Shared-camera/landmark Sim(3)'):
                aligned = align_sparse_sections(reference.model, local.model, alignment)
        except Exception as error:
            alignment.update(status='rejected', error=str(error))
            raise
        finally:
            write_json(self.output / 'alignment.json', alignment)
        # Root-level viewers share one RGB directory, checked against the saved
        # VGGT input independently by each child initializer.
        images = self.output / 'images'
        images.mkdir()
        for child in (reference, local):
            for frame, path in child.diagnostic.images.items():
                target = images / path.name
                if not target.exists():
                    shutil.copyfile(path, target)
        viewer = reference.diagnostic
        viewer.viewer(aligned, self.output / 'aligned', 'Local cluster aligned into reference gauge',
                      {'input_frames': len(aligned.cameras)})
        self.report['stage'] = 'reconciliation'
        reconciliation: dict = {}
        try:
            combined = merge_sparse_sections(reference.model, aligned, reconciliation)
        finally:
            write_json(self.output / 'reconciliation.json', reconciliation)
        combined.write(self.output / 'combined_sparse', sorted(combined.cameras))
        viewer.viewer(combined, self.output / 'before_joint', 'Combined sparse initialization',
                      {'input_frames': len(combined.cameras)})
        validation = MergeValidation.prepare(combined, reference.model, aligned)
        write_json(self.output / 'withheld_observations.json', validation.document())
        self.report['stage'] = 'joint_ba'
        self.publish()
        options = CuNLSOptions(use_gnc=False, check_jacobians=self.check_jacobians)
        refined, joint_report = CuNLSBundleAdjuster(self.device, options).optimize(
            validation.training, self.output / 'joint_ba')
        self.report['joint_ba'] = joint_report
        self.report['registered_frames'] = len(refined.cameras)
        checks = validation.evaluate(refined)
        write_json(self.output / 'validation.json', checks)
        self.report['validation'] = {key: value for key, value in checks.items() if key != 'observations'}
        if refined.points:
            viewer.viewer(refined, self.output / 'after_joint', 'cuNLS joint BA candidate — inspect validation', joint_report)
        passed = (checks['status'] == 'accepted' and joint_report['status'] == 'diagnostic_complete'
                  and set(refined.cameras) == (frame_sets[0] | frame_sets[1]))
        self.report.update(status='diagnostic_complete' if passed else 'diagnostic_rejected', stage='finished')
        self.publish()
        return self.report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Saved run with keyframes, processed images and global tracks')
    parser.add_argument('--reference', type=Path, help='Optional saved reference .pt; requires --local')
    parser.add_argument('--local', type=Path, help='Optional saved local .pt; requires --reference')
    parser.add_argument('--start', type=int, default=0, help='First selected-keyframe index, not original video frame')
    parser.add_argument('--cluster-size', type=int, default=32)
    parser.add_argument('--overlap', type=int, default=8)
    parser.add_argument('--checkpoint', type=Path, help='Override saved checkpoint when inferring new clusters')
    parser.add_argument('--device', type=int, default=0, help='Logical CUDA device; inference and BA run sequentially')
    parser.add_argument('--check-jacobians', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    diagnostic = None
    try:
        if (args.reference is None) != (args.local is None):
            raise ValueError('Supply both --reference and --local, or neither')
        if args.device < 0:
            raise ValueError('Device index must be nonnegative')
        CuNLSBundleAdjuster.preflight()
        run = args.run.expanduser().resolve()
        output = (args.output or run / datetime.now(timezone.utc).strftime('cunls_pair_%Y%m%d_%H%M%S_%f')).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        diagnostic = PairDiagnostic(run, output, args.device, args.check_jacobians)
        diagnostic.publish()
        sections = diagnostic.prepare(args.reference, args.local, args.start, args.cluster_size, args.overlap, args.checkpoint)
        report = diagnostic.execute(sections)
        LOGGER.info('%s: %d/%d pair cameras. Open %s', report['status'], report['registered_frames'],
                    report['input_frames'], output / 'index.html')
        return 0 if report['status'] == 'diagnostic_complete' else 1
    except (Exception, KeyboardInterrupt) as error:
        if diagnostic is not None:
            diagnostic.report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                     error=str(error) or 'Interrupted')
            diagnostic.publish()
        LOGGER.error('%s%s', str(error) or 'Interrupted',
                     f'\nArtifacts retained in {diagnostic.output}' if diagnostic else '')
        if args.debug:
            LOGGER.exception('Two-cluster cuNLS diagnostic traceback')
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
