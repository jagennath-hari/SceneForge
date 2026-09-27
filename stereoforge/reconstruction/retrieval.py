"""Isolated official SelaVPR++ inference; content-addressed descriptors only.

The worker keeps upstream's generic `model` imports out of the reconstruction
process. Its exit also releases all descriptor-model CUDA allocations.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from stereoforge.utils.progress import Progress


def file_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class GlobalDescriptors:
    def __init__(self, device: int) -> None:
        self.device = f'cuda:{device}'
        self.root = Path(os.environ.get('SELAVPR_ROOT', '/opt/third_party/SelaVPRplusplus'))
        if not (self.root / 'hubconf.py').is_file():
            raise FileNotFoundError('SelaVPR++ source is missing; rebuild the geometry Docker image')
        self.cache = Path(torch.hub.get_dir()).parent / 'stereoforge' / 'selavpr_descriptors'

    def extract(self, paths: list[Path], destination: Path) -> None:
        with Progress('Preparing SelaVPR++ descriptors'):
            model = torch.hub.load(str(self.root), 'SelaVPRplusplus', source='local',
                                  backbone='dinov2-base', aggregation='gem', hashing=False, rerank=False)
            model = model.module.eval()
            checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/SelaVPRplusplus_base.pth'
            try:
                xformers_version = version('xformers')
            except PackageNotFoundError:
                xformers_version = None
            attention = sys.modules.get('model.dinov2.attention')
            identity = {
                'checkpoint_sha256': file_digest(checkpoint),
                'source': {str(p.relative_to(self.root)): file_digest(p) for p in
                           sorted([self.root / 'hubconf.py', *(self.root / 'model').rglob('*.py')])},
                'preprocessing': 'RGB/ToTensor/ImageNet-normalize/322-square-bilinear-antialias-v1',
                'backbone': 'dinov2-base', 'aggregation': 'gem', 'hashing': False, 'rerank': False,
                'torch': torch.__version__, 'torchvision': torchvision.__version__, 'dtype': 'float32',
                'xformers': xformers_version,
                'xformers_attention': bool(getattr(attention, 'XFORMERS_AVAILABLE', False)),
            }
            namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            folder = self.cache / namespace
            folder.mkdir(parents=True, exist_ok=True)
        descriptors = np.empty((len(paths), 2048), dtype=np.float32)
        image_hashes = []
        pending = []
        on_device = False
        with Progress('Extracting global descriptors', len(paths), 'frame') as progress:
            def infer_pending() -> None:
                nonlocal on_device
                if not pending:
                    return
                if not on_device:
                    model.to(self.device)
                    on_device = True
                batch = torch.stack([item[2] for item in pending]).to(self.device)
                with torch.inference_mode():
                    values = model(batch).float().cpu().numpy()
                if values.shape != (len(pending), 2048) or not np.isfinite(values).all():
                    raise ValueError('SelaVPR++ returned invalid descriptors')
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                if np.any(norms < 1e-6):
                    raise ValueError('SelaVPR++ returned zero descriptors')
                values /= norms
                for (index, target, _), value in zip(pending, values, strict=True):
                    temporary = target.with_suffix('.' + uuid4().hex + '.partial')
                    with temporary.open('wb') as stream:
                        np.save(stream, value, allow_pickle=False)
                    temporary.replace(target)
                    descriptors[index] = value
                    progress.advance()
                pending.clear()

            for index, path in enumerate(paths):
                digest = file_digest(path)
                image_hashes.append(digest)
                target = folder / f'{digest}.npy'
                if target.is_file():
                    value = np.load(target, allow_pickle=False)
                    if value.shape != (2048,) or not np.isfinite(value).all() or abs(np.linalg.norm(value)-1) > 1e-3:
                        raise ValueError(f'Invalid cached descriptor: {target}')
                    descriptors[index] = value
                    progress.advance()
                    continue
                with Image.open(path) as image:
                    tensor = TF.to_tensor(image.convert('RGB'))
                tensor = TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225])
                tensor = TF.resize(tensor, [322, 322], interpolation=InterpolationMode.BILINEAR, antialias=True)
                pending.append((index, target, tensor))
                if len(pending) == 4:
                    infer_pending()
            infer_pending()
        temporary = destination.with_suffix('.partial')
        with temporary.open('wb') as stream:
            np.save(stream, descriptors, allow_pickle=False)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps({**identity, 'images': image_hashes}, indent=2))


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    GlobalDescriptors(request['device']).extract([Path(p) for p in request['paths']], Path(request['output']))
