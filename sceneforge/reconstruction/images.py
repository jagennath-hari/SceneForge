"""Bounded CPU preparation: decode once for geometry and global retrieval."""

from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sceneforge.video.sampling import VideoFrameSampler
from .retrieval import GlobalDescriptors


@dataclass(slots=True)
class PreparedImage:
    index: int
    path: Path
    size: tuple[int, int]
    key: str | None = None
    descriptor: np.ndarray | None = None
    retrieval_input: torch.Tensor | None = None


class FeatureImages:
    """Two workers and four queued frames overlap CPU work with descriptor inference."""

    def __init__(self, folder: Path, reused: dict[Path, Path], descriptors: GlobalDescriptors | None) -> None:
        self.folder, self.reused, self.descriptors = folder, reused, descriptors

    def prepare(self, item: tuple[int, Path]) -> PreparedImage:
        # Use the pinned upstream crop/shape helpers rather than a second geometry
        # convention. The RGB resize is exactly its balanced 512/16 path.
        from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio, _balanced_target_shape

        index, source = item
        destination = self.folder / f'{index:06d}.png'
        if not destination.exists() and source in self.reused:
            VideoFrameSampler._link_or_copy(self.reused[source], destination)
        key, descriptor = None, None
        payload = None
        if self.descriptors is not None:
            payload = source.read_bytes()
            key, descriptor = self.descriptors.lookup(payload)
        retrieval_input = None
        if not destination.exists() or (self.descriptors is not None and descriptor is None):
            with Image.open(BytesIO(payload) if payload is not None else source) as image:
                if image.mode == 'RGBA':
                    image = Image.alpha_composite(Image.new('RGBA', image.size, (255, 255, 255, 255)), image)
                rgb = image.convert('RGB')
            if not destination.exists():
                geometry = _crop_to_supported_aspect_ratio(rgb)
                height, width = _balanced_target_shape(geometry.height / max(geometry.width, 1), 512, 16)
                geometry = geometry.resize((width, height), Image.Resampling.BICUBIC)
                temporary = destination.with_suffix('.partial')
                geometry.save(temporary, format='PNG', compress_level=1)
                temporary.replace(destination)
            if self.descriptors is not None and descriptor is None:
                retrieval_input = self.descriptors.preprocess(rgb)
        with Image.open(destination) as image:
            size = image.size
        return PreparedImage(index, destination, size, key, descriptor, retrieval_input)

    def iterate(self, paths: list[Path]) -> Iterator[PreparedImage]:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='feature-images') as pool:
            pending = deque()
            source = iter(enumerate(paths))
            for item in source:
                pending.append(pool.submit(self.prepare, item))
                if len(pending) == 4:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
