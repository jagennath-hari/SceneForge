"""Scale-normalized surface refinement using Open3D tensor CPU/CUDA kernels."""

import logging
import re

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def refine_overlap(
    source: np.ndarray, target: np.ndarray, scale: float,
    rotation: np.ndarray, translation: np.ndarray, depth_unit: float,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Refine rigid alignment with fixed scale; validate geometry independently.

    Upload overlap clouds once. Downsampling, normals and ICP run on the selected
    device. DLPack shares surfaces with PyTorch for conditioning checks; only the
    6x6 normal matrix and small transforms/statistics return to the CPU.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Overlap ICP requires Open3D: uv pip install 'open3d>=0.19,<0.20'") from exc
    if device == "auto":
        device = "cuda:0" if o3d.core.cuda.is_available() else "cpu"
        if device == "cpu":
            LOGGER.warning("Open3D CUDA unavailable; overlap ICP is using CPU")
    if device != "cpu" and re.fullmatch(r"cuda:\d+", device) is None:
        raise ValueError("ICP device must be auto, cpu, or cuda:N")
    if device.startswith("cuda:") and (
        not o3d.core.cuda.is_available() or int(device.split(":")[1]) >= o3d.core.cuda.device_count()
    ):
        raise RuntimeError(f"Open3D cannot use {device}; check its CUDA build and visible GPUs")
    selected = o3d.core.Device("CPU:0" if device == "cpu" else device.upper())
    registration = o3d.t.pipelines.registration
    origin = np.median(target, axis=0)
    moved = (scale * (source @ rotation.T) + translation - origin) / depth_unit
    fixed = (target - origin) / depth_unit
    source_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(moved, o3d.core.float32, selected))
    target_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(fixed, o3d.core.float32, selected))
    transform = o3d.core.Tensor.eye(4, o3d.core.float64, o3d.core.Device("CPU:0"))
    levels = []

    def report(status: str) -> dict:
        return {"status": status, "device": str(selected), "levels": levels}

    for voxel in (0.08, 0.04, 0.02):
        a = source_cloud.voxel_down_sample(voxel)
        b = target_cloud.voxel_down_sample(voxel)
        if min(len(a.point.positions), len(b.point.positions)) < 30:
            return rotation, translation, report("insufficient_surface")
        b.estimate_normals(max_nn=30, radius=voxel * 3)
        kernel = registration.robust_kernel.RobustKernel(
            registration.robust_kernel.RobustKernelMethod.TukeyLoss, scaling_parameter=voxel)
        result = registration.icp(
            a, b, voxel * 3, transform,
            registration.TransformationEstimationPointToPlane(kernel),
            registration.ICPConvergenceCriteria(max_iteration=40),
        )
        matrix = result.transformation.cpu().numpy()
        # Tensor ICP returns one target index per source point, with -1 for no match.
        # Synchronize Open3D work before exposing allocations through legacy DLPack.
        if device.startswith("cuda:"):
            o3d.core.cuda.synchronize(selected)
        indices = torch.utils.dlpack.from_dlpack(result.correspondence_set.to_dlpack()).reshape(-1)
        positions = torch.utils.dlpack.from_dlpack(a.point.positions.to_dlpack())
        target_positions = torch.utils.dlpack.from_dlpack(b.point.positions.to_dlpack())
        target_normals = torch.utils.dlpack.from_dlpack(b.point.normals.to_dlpack())
        indices = indices.to(device=positions.device, dtype=torch.long)
        valid = indices >= 0
        count = int(valid.sum().item())
        if count < 30 or not np.isfinite(matrix).all():
            return rotation, translation, report("insufficient_correspondences")
        update = torch.as_tensor(matrix, dtype=positions.dtype, device=positions.device)
        points = positions[valid] @ update[:3, :3].T + update[:3, 3]
        normals = target_normals[indices[valid]]
        residuals = ((points - target_positions[indices[valid]]) * normals).sum(dim=1)
        weights = (1 - (residuals / voxel).square()).clamp_min(0).square()
        jacobian = torch.cat((torch.linalg.cross(points, normals, dim=1), normals), dim=1).double()
        weighted = jacobian * weights.sqrt()[:, None]
        eigenvalues = np.linalg.eigvalsh((weighted.T @ weighted).cpu().numpy())
        conditioned = bool(eigenvalues[-1] > 0 and eigenvalues[0] / eigenvalues[-1] > 1e-8)
        levels.append({"voxel_depth_units": voxel, "fitness": float(result.fitness),
                       "surface_rmse_depth_units": float(result.inlier_rmse),
                       "well_constrained": conditioned, "correspondences": count})
        if not conditioned:
            return rotation, translation, report("degenerate_surface")
        transform = result.transformation
    correction, offset = matrix[:3, :3], matrix[:3, 3]
    return (correction @ rotation,
            correction @ (translation - origin) + depth_unit * offset + origin,
            report("refined"))
