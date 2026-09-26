"""Route stage status and keyframe thumbnails to the Rust presentation adapter.

Unposed placement and independent-window offsets are visualization only.
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
        self.selected_previews: set[int] = set()
        self.dense_samples: list[tuple[np.ndarray, np.ndarray]] = []

    def _disable(self, error: Exception) -> None:
        self.enabled = False
        logging.getLogger(__name__).warning('Frontend visualization disabled: %s', error)

    def event(self, message: str) -> None:
        if self.enabled:
            try:
                self.native.builder.rerun_event(message)
            except Exception as error:
                self._disable(error)

    def activity(self, label: str, frames=(), pairs=(), *, ready: bool = False) -> None:
        self.event(json.dumps({"label": label, "frames": list(frames)[:64],
                               "pairs": list(pairs)[:16], "ready": ready}))

    def dense(self, xyz: np.ndarray, rgb: np.ndarray, *, final: bool = False) -> None:
        if not self.enabled:
            return
        try:
            if final:
                self.dense_samples.clear()
            else:
                stride = max(1, (len(xyz)+511)//512)
                self.dense_samples.append((xyz[::stride].copy(), rgb[::stride].copy()))
                # Bound retained samples and logging volume even for long sequences.
                self.dense_samples = self.dense_samples[-400:]
                xyz = np.concatenate([p for p, _ in self.dense_samples])
                rgb = np.concatenate([c for _, c in self.dense_samples])
            stride = max(1, (len(xyz)+239999)//240000)
            self.native.builder.rerun_dense(np.ascontiguousarray(xyz[::stride], dtype=np.float32),
                                           np.ascontiguousarray(rgb[::stride], dtype=np.uint8))
        except Exception as error:
            self._disable(error)

    def matched_batch(self, path: Path) -> None:
        # Use measured verification results, never mark requested pairs as passed.
        verified = []
        attempted = 0
        with path.open() as stream:
            for line in stream:
                pair = json.loads(line)
                attempted += 1
                matches = pair['matches']
                inliers = sum(bool(m['inlier']) for m in matches)
                if inliers >= 30 and inliers >= 0.25 * len(matches):
                    verified.append((pair['source'], pair['target']))
        sample = verified[:16]
        self.activity(f"Verified {len(verified)}/{attempted} pairs — sampled schematic connections",
                      sorted({f for pair in sample for f in pair}), sample, ready=True)

    def keyframe(self, frame: int, path: Path) -> None:
        if not self.enabled:
            return
        try:
            with Image.open(path) as source:
                image = source.convert('RGB')
                image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                pixels = np.array(image, dtype=np.uint8)
            self.native.builder.rerun_keyframe(frame, pixels)
        except Exception as error:
            self._disable(error)

    def selection(self, event: dict, image: Path | None = None) -> None:
        if event.get('event') == 'accepted_keyframe':
            frame = event['keyframe_index']
            self.selected_previews.add(frame)
            if image is not None:
                self.keyframe(frame, image)
            return
        now = monotonic()
        if now-self.last_selection < 0.5 and event['saved'] != event['total']:
            return
        self.last_selection = now
        self.event(f"{event.get('stage', 'Selecting keyframes')}: {event['saved']}/{event['total']} candidates. "
                   "Only accepted keyframes are placed on the schematic sphere.")

    def keyframes(self, paths: tuple[Path, ...]) -> None:
        self.event(f"Selected {len(paths)} keyframes — schematic sphere with display-only positions and FOV")
        for frame, path in enumerate(paths):
            if frame not in self.selected_previews:
                self.keyframe(frame, path)

    def window(self, path: Path, images: list[Path], identifier: int) -> None:
        if not self.enabled:
            return
        try:
            self.event(f"VGGT window {identifier} ready: separated display group, not yet aligned to the common map")
            frames = self.native._load_frames(path, images)
            self.native.builder.preview_window(list(frames.values()))
        except Exception as error:
            self._disable(error)
