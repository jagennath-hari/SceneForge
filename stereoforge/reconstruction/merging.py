"""Align a refined VGGT window with the common map and jointly refine it."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil

from stereoforge.refinement.cunls import CuNLSBundleAdjuster, CuNLSOptions
from stereoforge.refinement.sparse_model import SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from .cluster import ClusterInitializer
from .merge_validation import MergeValidation, observation_key
from .registration import align_sparse_sections, merge_sparse_sections, shared_landmarks



@dataclass(frozen=True, slots=True)
class RefinedCluster:
    model: SparseModel
    track_ids: tuple[int, ...]
    artifacts: ClusterInitializer


class WindowMerger:
    """Keep every stage on disk; never publish an unvalidated merge as complete."""

    def __init__(self, run: Path, output: Path, device: int, check_jacobians: bool,
                 lm_iterations: int = 200, diagnostics: bool = True) -> None:
        self.run, self.output, self.device = run, output, device
        self.check_jacobians = check_jacobians
        self.diagnostics = diagnostics
        if not 1 <= lm_iterations <= 500:
            raise ValueError('Joint LM budget must be between 1 and 500')
        self.lm_iterations = lm_iterations
        self.result: RefinedCluster | None = None
        self.validations: list[MergeValidation] = []
        self.report: dict = {
            'status': 'running', 'stage': 'preparing', 'input_run': str(run),
            'scope': 'Window-to-common-map alignment and joint sparse bundle adjustment',
            'backend': 'Custom C++/CUDA cuNLS local and joint BA; CPU Sim(3) registration',
        }

    def publish(self) -> None:
        write_json(self.output / 'report.json', self.report)

    def refine(self, name: str, section: Path) -> RefinedCluster:
        self.report['stage'] = f'{name}_local_ba'
        self.publish()
        directory = self.output / name
        directory.mkdir()
        diagnostic = ClusterInitializer(self.run, section, directory, self.device, self.check_jacobians, self.diagnostics)
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

    def merge(self, reference: RefinedCluster, local: RefinedCluster,
              previous_validations: tuple[MergeValidation, ...] = ()) -> dict:
        """Merge validated children, retaining identities and earlier boundary checks."""
        frame_sets = [set(reference.model.cameras), set(local.model.cameras)]
        shared = frame_sets[0] & frame_sets[1]
        if len(shared) < 6 or not frame_sets[0]-shared or not frame_sets[1]-shared:
            raise ValueError('Merge requires six shared cameras and distinct child frames')
        self.report.update(input_frames=len(frame_sets[0] | frame_sets[1]), shared_frames=sorted(shared))
        identities = {}
        for child in (reference, local):
            if len(child.track_ids) != len(child.model.points) or len(set(child.track_ids)) != len(child.track_ids):
                raise ValueError('Child landmark identity count mismatch')
            for point, track in zip(child.model.points, child.track_ids, strict=True):
                for frame, uv in point.observations.items():
                    key = observation_key(frame, uv)
                    if key in identities and identities[key] != track:
                        raise ValueError('Children assign the same image observation to different tracks')
                    identities[key] = track
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
        if self.diagnostics:
            images.mkdir()
            for child in (reference, local):
                for frame, path in child.artifacts.images.items():
                    target = images / path.name
                    if not target.exists():
                        shutil.copyfile(path, target)
        viewer = reference.artifacts
        viewer.export_diagnostics(aligned, self.output / 'aligned', 'Local cluster aligned into reference gauge',
                      {'input_frames': len(aligned.cameras)})
        self.report['stage'] = 'reconciliation'
        reconciliation: dict = {}
        support_audit: dict = {}

        def audit_support(stage: str, model: SparseModel) -> None:
            boundaries = []
            for check in previous_validations:
                result = check.evaluate(model)
                missing = [i for i, record in enumerate(result['observations']) if record['error_pixels'] is None]
                outside = [i for i, record in enumerate(result['observations'])
                           if record['error_pixels'] is not None and not record['passed']]
                boundaries.append({'shared_frames': list(check.shared_frames),
                                   'held_out_count': result['held_out_count'],
                                   'unusable_observation_indices': missing,
                                   'finite_error_failure_indices': outside,
                                   'within_5px': result['within_5px']})
            support_audit[stage] = boundaries
            write_json(self.output / 'validation_support.json', support_audit)

        # Observation indices refer to the immutable ancestor_holdouts records.
        for name, child in (('reference_before_reconciliation', reference.model),
                            ('local_before_reconciliation', aligned)):
            audit_support(name, child)
        try:
            # Stable global IDs also join tracks whose common observations
            # were removed by one child's local BA filtering.
            reference_ids = {track: i for i, track in enumerate(reference.track_ids)}
            track_pairs = [(reference_ids[track], j) for j, track in enumerate(local.track_ids)
                           if track in reference_ids]
            combined = merge_sparse_sections(reference.model, aligned, reconciliation, track_pairs=track_pairs)
        finally:
            write_json(self.output / 'reconciliation.json', reconciliation)
        audit_support('after_reconciliation', combined)
        # Earlier holdouts must not re-enter training through the other child.
        excluded = {observation_key(item['frame'], item['uv'])
                    for check in previous_validations for item in check.observations}
        retained_points = []
        for point in combined.points:
            observations = {f: uv for f, uv in point.observations.items()
                            if observation_key(f, uv) not in excluded}
            if len(observations) >= 3:
                retained_points.append(replace(point, observations=observations))
        combined = SparseModel(combined.cameras, tuple(retained_points))
        write_json(self.output / 'ancestor_holdouts.json', {
            'boundaries': [{**check.document(), 'training_frames': sorted(check.training.cameras)}
                           for check in previous_validations]})
        audit_support('after_ancestor_holdout_exclusion', combined)
        combined.write(self.output / 'combined_sparse', sorted(combined.cameras))
        viewer.export_diagnostics(combined, self.output / 'before_joint', 'Combined sparse initialization',
                      {'input_frames': len(combined.cameras)})
        validation = MergeValidation.prepare(combined, reference.model, aligned)
        audit_support('joint_training', validation.training)
        write_json(self.output / 'withheld_observations.json', validation.document())
        self.report['stage'] = 'joint_ba'
        self.publish()
        options = CuNLSOptions(use_gnc=False, check_jacobians=self.check_jacobians,
                               lm_iterations=self.lm_iterations, huber_delta_pixels=3.0)
        refined, joint_report = CuNLSBundleAdjuster(self.device, options).optimize(
            validation.training, self.output / 'joint_ba')
        audit_support('after_joint_ba', refined)
        self.report['joint_ba'] = joint_report
        self.report['registered_frames'] = len(refined.cameras)
        checks = validation.evaluate(refined)
        write_json(self.output / 'validation.json', checks)
        self.report['validation'] = {key: value for key, value in checks.items() if key != 'observations'}
        if refined.points:
            viewer.export_diagnostics(refined, self.output / 'after_joint', 'cuNLS joint BA candidate — inspect validation', joint_report)
        passed = (checks['status'] == 'accepted' and joint_report['status'] == 'diagnostic_complete'
                  and set(refined.cameras) == (frame_sets[0] | frame_sets[1]))
        ancestor_checks = [check.evaluate(refined) for check in previous_validations]
        write_json(self.output / 'ancestor_validation.json', {'boundaries': ancestor_checks})
        self.report['ancestor_boundaries_passed'] = [check['status'] == 'accepted' for check in ancestor_checks]
        passed = passed and all(self.report['ancestor_boundaries_passed'])
        track_ids = []
        for point in refined.points:
            votes = {identities.get(observation_key(frame, uv)) for frame, uv in point.observations.items()}
            if len(votes) != 1 or None in votes:
                raise ValueError('Merged point lacks an unambiguous global track identity')
            track_ids.append(votes.pop())
        if len(set(track_ids)) != len(track_ids):
            raise ValueError('Merged model contains duplicate global tracks')
        write_json(self.output / 'optimized_track_ids.json', {'point_track_ids': track_ids})
        if passed:
            merged_viewer = ClusterInitializer(self.run, self.output, self.output, self.device, diagnostics=self.diagnostics)
            merged_viewer.images = ({frame: images / f'{frame:06d}.png' for frame in refined.cameras}
                                    if self.diagnostics else {**reference.artifacts.images, **local.artifacts.images})
            self.result = RefinedCluster(refined, tuple(track_ids), merged_viewer)
            # Earlier-boundary evaluation needs camera identities and withheld
            # observation keys, not another copy of every historical map point.
            compact_validation = replace(validation, training=SparseModel(validation.training.cameras, ()))
            self.validations = [*previous_validations, compact_validation]
        self.report.update(status='diagnostic_complete' if passed else 'diagnostic_rejected', stage='finished')
        self.publish()
        return self.report
