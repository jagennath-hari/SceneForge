"""Reuse saved predictions, then retry one smaller inference level at a stalled frontier."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import progress_group
from .inference import infer_clusters
from .native_map import WindowRejected
from .view_graph import Cluster

if TYPE_CHECKING:
    from .windowed import WindowReconstructor


@dataclass(slots=True)
class WindowCandidate:
    key: str
    parent: int
    frames: tuple[int, ...]
    source: Path
    kind: str
    attempted_support: set[frozenset[int]] = field(default_factory=set)
    waiting_for: frozenset[int] = frozenset()
    blocked: bool = False
    accepted: bool = False
    reason: str | None = None
    supported_subset_proposed: bool = False


class WindowRecovery:
    """All candidates use the same transactional robust BA and graph-support checks.

    Saved subsets are not new predictions: they provide a possible bridge to a
    neighboring full prediction. Fresh smaller predictions are attempted only
    after available saved candidates cannot extend the accepted map. Recovery
    children never trigger another inference level. A failed initialization can
    propose one supported subset per candidate, without recursively pruning it.
    """

    def __init__(self, reconstruction: WindowReconstructor, windows: list[Cluster],
                 paths: list[Path], checkpoint: Path) -> None:
        self.reconstruction = reconstruction
        self.windows = windows
        self.paths = paths
        self.checkpoint = checkpoint
        self.owners: dict[int, Path] = {}
        self.saved_subsets: set[int] = set()
        self.split_predictions: set[int] = set()
        self.candidates = [WindowCandidate(str(window.identifier), window.identifier,
            tuple(window.frames), reconstruction.output / 'vggt' / f'{window.identifier}.pt', 'original')
            for window in windows]
        self.events: list[dict] = []

    @staticmethod
    def _subwindows(frames: list[int]) -> list[tuple[int, ...]]:
        if len(frames) <= 8:
            return []
        size = max(8, (len(frames)+1)//2)
        overlap = max(6, size//2)
        stride = size-overlap
        starts = list(range(0, len(frames)-size+1, stride))
        if starts[-1] != len(frames)-size:
            starts.append(len(frames)-size)
        return [tuple(frames[start:start+size]) for start in starts]

    def _eligible(self) -> list[WindowCandidate]:
        owned = self.owners.keys()
        return [candidate for candidate in self.candidates
                if not candidate.accepted and not candidate.blocked
                and not set(candidate.frames).issubset(owned)
                and (not candidate.waiting_for or not candidate.waiting_for.isdisjoint(owned))
                and frozenset(set(candidate.frames) & owned) not in candidate.attempted_support]

    def _propose_supported_subset(self, candidate: WindowCandidate, frames: tuple[int, ...]) -> dict | None:
        if (not frames or candidate.kind == 'supported_subset' or
                candidate.supported_subset_proposed):
            return None
        selected = set(frames)
        original = set(candidate.frames)
        excluded = sorted(original-selected)
        # Match the native bound and reject malformed proposals rather than
        # treating arbitrary frame omission as successful reconstruction.
        if (len(selected) != len(frames) or not selected < original or len(selected) < 8 or
                len(excluded) > min(8, len(original)//4)):
            raise ValueError('Native supported-subset proposal violates recovery bounds')
        candidate.supported_subset_proposed = True
        if selected.issubset(self.owners):
            return None
        selected_frames = tuple(sorted(selected))
        if any(other.source == candidate.source and other.frames == selected_frames for other in self.candidates):
            return None
        key = f'{candidate.key}/supported'
        self.candidates.append(WindowCandidate(key, candidate.parent, selected_frames,
                                               candidate.source, 'supported_subset'))
        logging.info('Window %s recovery: proposing %d/%d supported cameras; omitted frames %s remain pending unless already registered',
                     candidate.key, len(frames), len(original), excluded)
        return {'candidate': key, 'frames': list(selected_frames), 'omitted_frames': excluded,
                'pending_omitted_frames': sorted(set(excluded)-self.owners.keys())}

    def _frontiers(self) -> list[Cluster]:
        owned = self.owners.keys()
        return sorted((window for window in self.windows
                       if not set(window.frames).issubset(owned)
                       and (not owned or not set(window.frames).isdisjoint(owned))
                       and self._subwindows(window.frames)),
                      key=lambda window: (-len(set(window.frames) & owned), window.identifier))

    def _expand(self, progress) -> bool:
        frontiers = self._frontiers()
        # Try all reusable bridges before spending any additional GPU inference.
        for window in frontiers:
            if window.identifier in self.saved_subsets:
                continue
            self.saved_subsets.add(window.identifier)
            groups = self._subwindows(window.frames)
            source = self.reconstruction.output / 'vggt' / f'{window.identifier}.pt'
            for index, frames in enumerate(groups):
                self.candidates.append(WindowCandidate(f'{window.identifier}/saved/{index}',
                    window.identifier, frames, source, 'saved_subset'))
            logging.info('Window %d recovery: trying %d overlapping subsets of its saved prediction',
                         window.identifier, len(groups))
            self._report()
            return True
        for window in frontiers:
            if window.identifier in self.split_predictions:
                continue
            self.split_predictions.add(window.identifier)
            groups = self._subwindows(window.frames)
            directory = self.reconstruction.output / 'recovery_vggt' / f'window_{window.identifier:04d}'
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {'parent': window.identifier, 'frames': [list(group) for group in groups]}
            manifest_path = directory / 'windows.json'
            if manifest_path.exists():
                if json.loads(manifest_path.read_text()) != manifest:
                    raise ValueError('Recovery prediction cache has different frame IDs')
            write_json(manifest_path, manifest)
            clusters = [Cluster(index, list(frames)) for index, frames in enumerate(groups)]
            logging.info('Window %d recovery: predicting %d smaller overlapping windows (one level only)',
                         window.identifier, len(groups))
            progress.status(f'window {window.identifier} recovery | smaller VGGT predictions')
            visualization = self.reconstruction.visualization

            def ready(identifier: int) -> None:
                if visualization is not None:
                    visualization.window(directory / f'{identifier}.pt',
                                         list(self.reconstruction.images.values()), identifier)

            infer_clusters(self.checkpoint, self.paths, clusters, directory,
                           self.reconstruction.devices, on_complete=ready)
            for index, frames in enumerate(groups):
                self.candidates.append(WindowCandidate(f'{window.identifier}/split/{index}',
                    window.identifier, frames, directory / f'{index}.pt', 'smaller_prediction'))
            self._report()
            return True
        return False

    def _report(self) -> None:
        owned = self.owners.keys()
        unresolved = [{'window': window.identifier, 'frames': window.frames,
                       'missing_frames': sorted(set(window.frames)-owned),
                       'reason': self.candidates[window.identifier].reason or 'No accepted bridge to this window'}
                      for window in self.windows if not set(window.frames).issubset(owned)]
        self.reconstruction.summary.update(
            unresolved_windows=unresolved,
            covered_windows=len(self.windows)-len(unresolved),
            recovery={'saved_subset_parents': sorted(self.saved_subsets),
                      'smaller_prediction_parents': sorted(self.split_predictions),
                      'attempts': self.events})
        write_json(self.reconstruction.attempt / 'window_recovery.json', {
            'registered_frames': len(self.owners), 'input_frames': len(self.paths),
            'unresolved_windows': unresolved, 'attempts': self.events})

    def run(self) -> dict[int, Path]:
        native = self.reconstruction.native
        summary = self.reconstruction.summary
        summary.update(merge_order=[], unresolved_windows=[], global_ba_complete=False)
        try:
            # Count unique accepted frames: replacement windows must not inflate
            # the original window total or mark an unresolved region complete.
            with progress_group('Building common map', len(self.paths), 'frame') as progress:
                while len(self.owners) < len(self.paths):
                    eligible = self._eligible()
                    ranked = native.rank_windows([list(candidate.frames) for candidate in eligible])
                    if not ranked:
                        if self._expand(progress):
                            continue
                        break
                    candidate = eligible[ranked[0]]
                    summary['stage'] = f'window_{candidate.key}'
                    support = frozenset(set(candidate.frames) & self.owners.keys())
                    candidate.attempted_support.add(support)
                    references = {frame: self.owners[frame] for frame in support}
                    stage = ''
                    bridge_recovery = False

                    def update(value: str) -> None:
                        nonlocal stage, bridge_recovery
                        stage = value
                        bridge_recovery |= value.startswith('initialization bridge')
                        progress.status(f'window {candidate.key} | {candidate.kind} | {value}')

                    event = {'candidate': candidate.key, 'kind': candidate.kind,
                             'frames': list(candidate.frames), 'map_revision': native.accepted_windows}
                    try:
                        native.add_window(candidate.source, self.reconstruction.images, references,
                                          update, frame_ids=candidate.frames)
                    except WindowRejected as error:
                        candidate.reason = str(error)
                        candidate.waiting_for = error.unanchored_frames
                        # A bridge chooses its reference component using shared
                        # accepted cameras. Retry it only when that support changes.
                        # Unrecovered local computations remain map-independent.
                        candidate.blocked = (not bridge_recovery and
                            (stage == 'initializing landmarks' or stage.startswith('local cuNLS BA:')))
                        event.update(status='rejected', stage=stage, reason=str(error))
                        proposal = self._propose_supported_subset(candidate, error.supported_frames)
                        if proposal is not None:
                            event['supported_subset_proposal'] = proposal
                        self.events.append(event)
                        logging.warning('Deferred window %s: %s', candidate.key, error)
                        self._report()
                        continue
                    new_frames = 0
                    for frame in candidate.frames:
                        if frame not in self.owners:
                            self.owners[frame] = candidate.source
                            new_frames += 1
                    candidate.accepted = True
                    event.update(status='accepted', added_frames=new_frames)
                    self.events.append(event)
                    summary['merge_order'].append(candidate.key)
                    summary['accepted_windows'] = native.accepted_windows
                    progress.advance(new_frames)
                    self._report()
        finally:
            self._report()
        return self.owners
