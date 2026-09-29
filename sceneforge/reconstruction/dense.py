# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jagennath Hari
#
# This file is part of SceneForge.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map-scaled VGGT-Ω depth, CUDA multi-view consistency refinement and native fusion.

Unsupported pixels retain their calibrated prior but never enter the dense cloud.
This is geometric consensus refinement, not a learned densifier or a metric map.
"""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import math

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as functional

from sceneforge.geometry.storage import load_sequence
from .models import SparseModel
from sceneforge.utils.progress import Progress
from .native_map import native_backend


@dataclass(frozen=True, slots=True)
class DenseOptions:
    relative_depth_tolerance: float = 0.05
    cycle_pixels: float = 2.0
    minimum_support: int = 2
    minimum_parallax_degrees: float = 0.5
    color_tolerance: float = 0.15
    voxel_depth_fraction: float = 0.01


class DenseRefiner:
    def __init__(self, output: Path, device: int, options: DenseOptions | None = None) -> None:
        self.output = output
        self.device = torch.device(f'cuda:{device}')
        self.options = options or DenseOptions()
        self.folder = output / 'dense'
        self.folder.mkdir(exist_ok=True)
        self._cache: OrderedDict[int, tuple] = OrderedDict()

    def _load(self, frame: int) -> tuple:
        if frame not in self._cache:
            with np.load(self.folder / f'{frame:06d}_prior.npz') as data:
                self._cache[frame] = tuple(torch.as_tensor(data[key].copy(), device=self.device)
                                           for key in ('depth', 'valid', 'rgb'))
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(frame)
        return self._cache[frame]

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        # Camera arrays can be immutable views. PyTorch first wraps NumPy
        # storage even when transferring to CUDA; do not expose read-only memory.
        if not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _measure(self, world: torch.Tensor, pixels: torch.Tensor, rgb: torch.Tensor,
                 reference, neighbor, neighbor_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        depth, valid, color = self._load(neighbor_id)
        h, w = depth.shape
        rn, cn = self._tensor(neighbor.pose[:3, :3]), self._tensor(neighbor.pose[:3, 3])
        rr, cr = self._tensor(reference.pose[:3, :3]), self._tensor(reference.pose[:3, 3])
        kn, kr = self._tensor(neighbor.intrinsics), self._tensor(reference.intrinsics)
        camera = (world-cn) @ rn
        projected = camera @ kn.T
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
        grid = 2*uv / uv.new_tensor([w-1, h-1])-1
        grid = torch.nan_to_num(grid, nan=2, posinf=2, neginf=-2)
        # Nearest sampling avoids interpolating depths across silhouette edges.
        channels = torch.cat((depth[..., None], valid[..., None], color), dim=-1)
        sampled = functional.grid_sample(channels.permute(2, 0, 1)[None].float(), grid[None],
                                         mode='nearest', padding_mode='zeros', align_corners=True)[0].permute(1, 2, 0)
        z = sampled[..., 0]
        relative = (camera[..., 2]-z).abs() / z.clamp_min(1e-8)
        rays = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1) @ torch.linalg.inv(kn).T
        back_world = (rays*z[..., None]) @ rn.T+cn
        back_camera = (back_world-cr) @ rr
        reprojection = back_camera @ kr.T
        back_uv = reprojection[..., :2]/reprojection[..., 2:].clamp_min(1e-8)
        cycle = torch.linalg.vector_norm(back_uv-pixels, dim=-1)
        a = functional.normalize(world-cr, dim=-1)
        b = functional.normalize(world-cn, dim=-1)
        parallax = (a*b).sum(-1) < math.cos(math.radians(self.options.minimum_parallax_degrees))
        agreement = ((camera[..., 2] > 0) & (back_camera[..., 2] > 0) & (sampled[..., 1] > 0.5)
                     & (uv[..., 0] >= 0) & (uv[..., 0] <= w-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
                     & (relative <= self.options.relative_depth_tolerance) & (cycle <= self.options.cycle_pixels)
                     & ((sampled[..., 2:]-rgb).abs().mean(-1) <= self.options.color_tolerance) & parallax)
        weight = torch.where(agreement, 1/(1+(relative/0.02).square()), 0.0)
        return torch.nan_to_num(back_camera[..., 2], nan=0, posinf=0, neginf=0), weight

    @torch.inference_mode()
    def run(self, model: SparseModel, owners: dict[int, Path], visualization=None) -> dict:
        observations: dict[int, list] = {frame: [] for frame in model.cameras}
        for point in model.points:
            for frame, pixel in point.observations.items():
                if frame in observations:
                    observations[frame].append((pixel, point.xyz))
        frames = sorted(model.cameras)
        anchored, rejected, scene_depths = [], [], []
        source_path, source_frames, source_rgb = None, {}, None
        with Progress('Calibrating dense depth', total=len(frames), unit='frame') as progress:
            for frame_index, frame in enumerate(frames):
                if visualization is not None:
                    visualization.activity(f"Calibrating dense depth: {frame_index+1}/{len(frames)}", [frame], follow_frame=frame, follow_start=frame_index == 0)
                path = owners[frame]
                if path != source_path:
                    sequence = load_sequence(path)
                    source_frames = {f.frame_index: (index, f) for index, f in enumerate(sequence.frames)}
                    source_rgb = sequence.processed_rgb
                    if source_rgb is None:
                        raise ValueError(f"VGGT-Ω window has no processed RGB: {path}")
                    source_path = path
                index, raw = source_frames[frame]
                depth = raw.depth.numpy()
                confidence = raw.confidence.numpy()
                camera = model.cameras[frame]
                valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
                if valid.any():
                    valid &= confidence >= np.quantile(confidence[valid], 0.3)
                pairs = observations[frame]
                ratios, zvalues = [], []
                for pixel, xyz in pairs:
                    x, y = np.rint(pixel).astype(int)
                    if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]) or not valid[y, x]:
                        continue
                    point = camera.pose[:3, :3].T @ (xyz-camera.pose[:3, 3])
                    if not np.isfinite(point).all() or point[2] <= 0:
                        continue
                    projected = camera.intrinsics @ point
                    if np.linalg.norm(projected[:2]/projected[2]-pixel) > 3:
                        continue
                    ratios.append(np.log(point[2]/depth[y, x])); zvalues.append(point[2])
                if len(ratios) < 6:
                    rejected.append({'frame': frame, 'reason': 'fewer than six reliable sparse depth anchors'})
                    progress.advance(); continue
                log_scale = float(np.median(ratios))
                if float(np.median(np.abs(np.asarray(ratios)-log_scale))) > 0.35:
                    rejected.append({'frame': frame, 'reason': 'inconsistent sparse-to-dense depth scale'})
                    progress.advance(); continue
                scale = np.exp(log_scale)
                scaled = np.where(valid, depth*scale, 0).astype(np.float32)
                if not np.isfinite(scaled).all():
                    rejected.append({'frame': frame, 'reason': 'nonfinite calibrated depth'})
                    progress.advance(); continue
                rgb = source_rgb[index].numpy().transpose(1, 2, 0).astype(np.float32)
                np.savez_compressed(self.folder / f'{frame:06d}_prior.npz', depth=scaled, valid=valid, rgb=rgb,
                                    scale=scale, intrinsics=camera.intrinsics, camera_to_world=camera.pose)
                anchored.append(frame); scene_depths.append(float(np.median(zvalues)))
                progress.advance()
        if not anchored:
            raise RuntimeError('Dense refinement has no map-calibrated depth frames')
        voxel_size = self.options.voxel_depth_fraction*float(np.median(scene_depths))
        fusion = native_backend().DenseFusion(voxel_size)
        anchored_set = set(anchored)
        accepted_pixels = 0
        coverage = []
        with Progress('Refining and fusing dense depth', total=len(anchored), unit='frame') as progress:
            for frame_index, frame in enumerate(anchored):
                if visualization is not None and frame_index == 0:
                    visualization.activity(f"Refining dense depth: 1/{len(anchored)}", [frame],
                                           follow_frame=frame, follow_start=True)
                camera = model.cameras[frame]
                prior, valid, rgb = self._load(frame)
                h, w = prior.shape
                y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
                pixels = torch.stack((x, y), dim=-1).float()
                rays = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1) @ torch.linalg.inv(self._tensor(camera.intrinsics)).T
                rotation, center = self._tensor(camera.pose[:3, :3]), self._tensor(camera.pose[:3, 3])
                world = (rays*prior[..., None]) @ rotation.T+center
                neighbors = [frame+offset for offset in (-4, -2, -1, 1, 2, 4) if frame+offset in anchored_set]
                inverse_sum, weight_sum = 1/prior.clamp_min(1e-8), torch.ones_like(prior)
                for neighbor in neighbors:
                    z, weight = self._measure(world, pixels, rgb, camera, model.cameras[neighbor], neighbor)
                    inverse_sum += weight/z.clamp_min(1e-8)
                    weight_sum += weight
                candidate = weight_sum/inverse_sum.clamp_min(1e-8)
                candidate_world = (rays*candidate[..., None]) @ rotation.T+center
                support = torch.zeros_like(prior, dtype=torch.uint8)
                # Validate the updated depth, not only the starting prior.
                for neighbor in neighbors:
                    _, weight = self._measure(candidate_world, pixels, rgb, camera, model.cameras[neighbor], neighbor)
                    support += (weight > 0).to(torch.uint8)
                accepted = valid & (support >= self.options.minimum_support) & torch.isfinite(candidate) & (candidate > 0)
                refined = torch.where(accepted, candidate, prior)
                count = int(accepted.sum().item()); accepted_pixels += count
                depth_cpu = refined.cpu().numpy()
                mask_cpu = accepted.cpu().numpy()
                depth_median = float(np.median(depth_cpu[mask_cpu])) if count else None
                coverage.append({'frame': frame, 'supported_pixels': count, 'total_pixels': h*w, 'depth_median': depth_median})
                preview_pixels = np.zeros((h,w),dtype=np.uint8)
                if count:
                    low, high = np.percentile(depth_cpu[mask_cpu],[2,98])
                    preview_pixels[mask_cpu] = (1+254*np.clip((depth_cpu[mask_cpu]-low)/max(float(high-low),1e-8),0,1)).astype(np.uint8)
                Image.fromarray(preview_pixels).save(self.folder / f'{frame:06d}_depth.png')
                Image.fromarray((mask_cpu*255).astype(np.uint8)).save(self.folder / f'{frame:06d}_support.png')
                np.savez_compressed(self.folder / f'{frame:06d}.npz', depth=refined.cpu().numpy(),
                                    supported=accepted.cpu().numpy(), supporting_views=support.cpu().numpy(),
                                    intrinsics=camera.intrinsics, camera_to_world=camera.pose)
                accepted_xyz = candidate_world[accepted].contiguous().cpu().numpy()
                accepted_rgb = (rgb[accepted]*255).round().clamp(0, 255).to(torch.uint8).contiguous().cpu().numpy()
                fusion.add(accepted_xyz, accepted_rgb, frame)
                if visualization is not None:
                    visualization.dense(accepted_xyz, accepted_rgb)
                    visualization.activity(
                        f"Dense refinement: {frame_index+1}/{len(anchored)}; {count} supported pixels — sampled preview",
                        [frame], ready=True, follow_frame=frame)
                progress.advance()
        if not accepted_pixels:
            raise RuntimeError('No dense pixels passed multi-view validation; sparse map retained')
        if visualization is not None:
            visualization.event('Writing final voxel-fused dense cloud')
        preview = fusion.write(str(self.output / 'dense_point_cloud.ply'))
        np.savez(self.folder / 'preview.npz', xyz=preview['xyz'], rgb=preview['rgb'], frames=preview['frames'])
        if visualization is not None:
            visualization.dense(preview['xyz'], preview['rgb'], final=True)
            visualization.activity(f"Dense reconstruction complete: {int(preview['count']):,} fused points; bounded viewer preview", final_scene=True)
        self._cache.clear()
        return {'method': 'map-scaled VGGT-Ω + inverse-depth multi-view consensus', 'anchored_frames': len(anchored),
                'unanchored_frames': rejected, 'coverage': coverage, 'supported_pixels': accepted_pixels,
                'fused_points': int(preview['count']), 'voxel_size': voxel_size,
                'unsupported_pixels': 'calibrated prior retained, excluded from cloud',
                'cloud': 'dense_point_cloud.ply', 'units': 'reconstruction_units'}
