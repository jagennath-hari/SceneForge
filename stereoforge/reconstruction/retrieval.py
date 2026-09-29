"""Cached SelaVPR++ descriptors and bounded, temporally separated loop proposals."""

from dataclasses import asdict, dataclass
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as transforms


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    batch_size: int = 8
    candidates_per_frame: int = 3
    minimum_frame_gap: int = 32
    minimum_time_gap: float = 10.0
    region_spacing_seconds: float = 5.0


class GlobalDescriptors:
    """One descriptor per immutable image; load the GPU model only on cache misses."""

    checkpoint_name = 'SelaVPRplusplus_base.pth'
    checkpoint_url = ('https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/'
                      'SelaVPR%2B%2B/SelaVPRplusplus_base.pth')
    dimension = 2048

    def __init__(self, device: int, options: RetrievalOptions = RetrievalOptions()) -> None:
        self.device = torch.device(f'cuda:{device}')
        self.options = options
        self.model = None
        self.root = Path(os.environ.get('SELAVPR_ROOT', '/opt/third_party/SelaVPRplusplus'))
        if not (self.root / 'hubconf.py').is_file():
            raise FileNotFoundError('SelaVPR++ is missing. Rebuild Docker with third_party/SelaVPRplusplus')
        checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / self.checkpoint_name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not checkpoint.is_file():
                temporary = checkpoint.with_name(checkpoint.name + '.partial_' + uuid4().hex)
                try:
                    torch.hub.download_url_to_file(self.checkpoint_url, str(temporary), progress=False)
                    temporary.replace(checkpoint)
                finally:
                    temporary.unlink(missing_ok=True)
        with checkpoint.open('rb') as stream:
            checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        code = hashlib.sha256()
        for path in sorted([self.root / 'hubconf.py', *(self.root / 'model').rglob('*.py')]):
            code.update(str(path.relative_to(self.root)).encode())
            code.update(path.read_bytes())
        try:
            attention_version = version('xformers')
        except PackageNotFoundError:
            attention_version = None
        self.identity = {
            'version': 1, 'backbone': 'dinov2-base', 'aggregation': 'gem',
            'hashing': False, 'rerank': False, 'checkpoint_sha256': checkpoint_hash,
            'implementation_sha256': code.hexdigest(), 'resize_hw': [322, 322],
            'normalization_mean': [0.485, 0.456, 0.406],
            'normalization_std': [0.229, 0.224, 0.225],
            'preprocessing': 'RGB; ToTensor; Normalize; bilinear hard_resize, antialias=True',
            'precision': 'float32', 'torch': torch.__version__,
            'torchvision': version('torchvision'), 'xformers': attention_version,
        }
        fingerprint = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'stereoforge/retrieval' / fingerprint
        self.cache.mkdir(parents=True, exist_ok=True)

    def lookup(self, image_bytes: bytes) -> tuple[str, np.ndarray | None]:
        key = hashlib.sha256(image_bytes).hexdigest()
        path = self.cache / f'{key}.npy'
        try:
            value = np.load(path, allow_pickle=False)
            if (value.shape == (self.dimension,) and value.dtype == np.float32
                    and np.isfinite(value).all() and abs(float(np.linalg.norm(value))-1) < 1e-4):
                return key, value
        except (OSError, ValueError, EOFError):
            pass
        return key, None

    @staticmethod
    def preprocess(rgb: Image.Image) -> torch.Tensor:
        # Match upstream datasets_ws.base_transform + hard_resize. This image is
        # separate from the VGGT-Ω/local-feature grid; no observation is rescaled.
        value = transforms.to_tensor(rgb)
        value = transforms.normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return transforms.resize(value, [322, 322], antialias=True)

    def infer(self, batch: list[tuple[str, torch.Tensor]]) -> np.ndarray:
        if self.model is None:
            # Upstream uses absolute imports under the generic name `model`.
            # Refuse a conflicting module instead of silently loading another one.
            loaded = sys.modules.get('model.network')
            namespace = sys.modules.get('model')
            wrong_module = loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(self.root.resolve())
            wrong_namespace = (loaded is None and namespace is not None and
                               str(self.root.resolve() / 'model') not in list(getattr(namespace, '__path__', ())))
            if wrong_module or wrong_namespace:
                raise RuntimeError('SelaVPR++ cannot load: a different package owns the model namespace')
            model = torch.hub.load(str(self.root), 'SelaVPRplusplus', source='local',
                                   backbone='dinov2-base', aggregation='gem', hashing=False, rerank=False)
            # The Hub wrapper is DataParallel; this adapter schedules one device explicitly.
            self.model = model.module.to(self.device).eval()
        with torch.inference_mode():
            values = self.model(torch.stack([value for _, value in batch]).to(self.device))
            if values.shape != (len(batch), self.dimension) or not torch.isfinite(values).all():
                raise ValueError('SelaVPR++ returned invalid global descriptors')
            norms = torch.linalg.vector_norm(values.float(), dim=1, keepdim=True)
            if torch.any(norms <= 1e-8):
                raise ValueError('SelaVPR++ returned a zero-length descriptor')
            result = (values.float() / norms).cpu().numpy()
        for (key, _), value in zip(batch, result, strict=True):
            destination = self.cache / f'{key}.npy'
            temporary = destination.with_suffix('.partial_' + uuid4().hex)
            try:
                with temporary.open('wb') as stream:
                    np.save(stream, value, allow_pickle=False)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return result

    def close(self) -> None:
        self.model = None
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def retrieve(self, descriptors: np.ndarray, timestamps: tuple[float, ...], neighbors: int) -> dict[tuple[int, int], float]:
        """Cosine ranking in bounded GPU blocks; stable CPU tie-break by frame ID."""
        count = len(timestamps)
        if descriptors.shape != (count, self.dimension) or not np.isfinite(descriptors).all():
            raise ValueError('Descriptor/frame count mismatch or nonfinite descriptors')
        times = np.asarray(timestamps, dtype=np.float64)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Loop retrieval requires strictly increasing video timestamps')
        indices = np.arange(count)
        candidates: dict[tuple[int, int], float] = {}
        with torch.inference_mode():
            matrix = torch.from_numpy(descriptors).to(self.device)
            for start in range(0, count, 128):
                scores = (matrix[start:start+128] @ matrix.T).cpu().numpy()
                for offset, row in enumerate(scores):
                    query = start + offset
                    eligible = indices[(np.abs(indices-query) > max(neighbors, self.options.minimum_frame_gap))
                                       & (np.abs(times-times[query]) >= self.options.minimum_time_gap)]
                    order = eligible[np.lexsort((eligible, -row[eligible]))]
                    selected: list[int] = []
                    for target in order:
                        target = int(target)
                        if any(abs(times[target]-times[other]) < self.options.region_spacing_seconds for other in selected):
                            continue
                        pair = (min(query, target), max(query, target))
                        candidates[pair] = float(row[target])
                        selected.append(target)
                        if len(selected) == self.options.candidates_per_frame:
                            break
        return dict(sorted(candidates.items()))

    def document(self) -> dict:
        return {'model': self.identity, 'retrieval': asdict(self.options)}


def loop_pair_quality(record: dict, width: int, height: int) -> dict:
    """Verify local geometry, not descriptor score; reject concentrated matches."""
    matches = record['matches']
    inliers = [match for match in matches if match['inlier']]
    cells = []
    for name in ('a', 'b'):
        pixels = np.asarray([match[name] for match in inliers], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(pixels).all():
            raise ValueError('Loop matcher returned nonfinite pixels')
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        grid = np.floor(pixels[valid] * [4/width, 4/height]).astype(np.int32)
        cells.append(len(np.unique(grid, axis=0)))
    return {'accepted': len(inliers) >= 30 and len(inliers) >= 0.25*len(matches) and min(cells) >= 4,
            'inliers': len(inliers), 'matches': len(matches), 'grid_cells': cells}
