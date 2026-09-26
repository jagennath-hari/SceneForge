"""Route stage status and keyframe thumbnails to the Rust presentation adapter.

Unposed placement and independent-window offsets are visualization only.
"""
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

    def _disable(self, error: Exception) -> None:
        self.enabled = False
        logging.getLogger(__name__).warning('Frontend visualization disabled: %s', error)

    def event(self, message: str) -> None:
        if self.enabled:
            try:
                self.native.builder.rerun_event(message)
            except Exception as error:
                self._disable(error)

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
                   "Only accepted keyframes are placed on the schematic grid.")

    def keyframes(self, paths: tuple[Path, ...]) -> None:
        self.event(f"Selected {len(paths)} keyframes — schematic grid with display-only positions and FOV")
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
