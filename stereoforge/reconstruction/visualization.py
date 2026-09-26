"""Route existing frontend outputs into StereoForge's Rust visualization adapter.

This module does no Rerun logging itself and never participates in reconstruction
acceptance. Images/features are bounded previews; the Rust SDK owns presentation.
"""
import json
import logging
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image


class ReconstructionVisualization:
    def __init__(self, native) -> None:
        self.native = native
        self.enabled = True
        self.last_selection = 0.0

    def _disable(self, error: Exception) -> None:
        self.enabled = False
        logging.getLogger(__name__).warning('Frontend visualization disabled: %s', error)

    def event(self, message: str, image: Path | np.ndarray | None = None,
              points=None, segments=None) -> None:
        if not self.enabled:
            return
        try:
            if isinstance(image, Path):
                with Image.open(image) as source:
                    pixels = np.array(source.convert('RGB'))
            elif image is None:
                pixels = np.empty((0, 0, 3), dtype=np.uint8)
            else:
                pixels = image
            xy = np.asarray([] if points is None else points, dtype=np.float32).reshape(-1, 2)
            lines = np.asarray([] if segments is None else segments, dtype=np.float32).reshape(-1, 4)
            self.native.builder.rerun_event(message, np.ascontiguousarray(pixels, dtype=np.uint8),
                                           np.ascontiguousarray(xy[:2000]), np.ascontiguousarray(lines[:300]))
        except Exception as error:
            self._disable(error)

    def selection(self, event: dict, image: Path | None = None) -> None:
        now = monotonic()
        if now-self.last_selection < 0.5 and event['saved'] != event['total']:
            return
        self.last_selection = now
        self.event(f"{event.get('stage', 'Selecting keyframes')}: {event['saved']}/{event['total']} candidates; "
                   f"source time={event.get('timestamp_seconds')}", image)

    def match_batch(self, path: Path, images: dict[int, Path]) -> None:
        if not self.enabled:
            return
        try:
            # One actual pair per batch, with verified inliers only. Never invent
            # matches or regenerate learned features merely for visualization.
            with path.open() as stream:
                pair = json.loads(next(stream))
            a, b = pair['source'], pair['target']
            with Image.open(images[a]) as source, Image.open(images[b]) as target:
                left, right = np.array(source.convert('RGB')), np.array(target.convert('RGB'))
            canvas = np.zeros((max(left.shape[0], right.shape[0]), left.shape[1]+right.shape[1], 3), dtype=np.uint8)
            canvas[:left.shape[0], :left.shape[1]] = left
            canvas[:right.shape[0], left.shape[1]:] = right
            matches = [m for m in pair['matches'] if m['inlier']]
            lines = [[*m['a'], m['b'][0]+left.shape[1], m['b'][1]] for m in matches[:300]]
            self.event(f"Verified matching batch {path.stem}: keyframes {a} → {b}; "
                       f"{len(matches)}/{len(pair['matches'])} RANSAC inliers (up to 300 displayed)", canvas, segments=lines)
        except Exception as error:
            self._disable(error)

    def tracks(self, tracks: list[dict], images: dict[int, Path]) -> None:
        if not self.enabled:
            return
        # Show real multi-view track observations in up to four image planes.
        selected = set(np.linspace(0, len(images)-1, min(4, len(images)), dtype=int).tolist())
        observations = {frame: [] for frame in selected}
        for track in tracks:
            for frame in selected.intersection(track):
                if len(observations[frame]) < 2000:
                    observations[frame].append(track[frame])
        for frame, points in sorted(observations.items()):
            self.event(f"Built {len(tracks)} measured tracks; observations in keyframe {frame} "
                       "(up to 2000 displayed; no 3D poses yet)", images[frame], points=points)

    def window(self, path: Path, images: list[Path], identifier: int) -> None:
        if not self.enabled:
            return
        try:
            self.event(f"VGGT window {identifier} ready: independent local coordinates, not aligned to the common map")
            frames = self.native._load_frames(path, images)
            self.native.builder.preview_window(list(frames.values()))
        except Exception as error:
            self._disable(error)
