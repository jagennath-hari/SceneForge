"""Refine a seven-parameter Sim(3) with landmark and shared-camera evidence."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from stereoforge.refinement.sparse_model import SparseModel


def refine_camera_alignment(
    initial: tuple[float, np.ndarray, np.ndarray],
    source: np.ndarray, target: np.ndarray,
    reference: SparseModel, local: SparseModel,
    shared: list[int], scene_scale: float, report: dict,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Use training landmarks only; hold out whole cameras from the objective.

    This changes only the section transform. Individual camera poses, intrinsics
    and landmark coordinates remain fixed within each section.
    """
    held = shared[::3]
    training = [f for f in shared if f not in held]
    if len(training) < 3 or len(held) < 2:
        raise ValueError("Camera-aware alignment needs three training and two held-out cameras")
    scale0, rotation0, translation0 = initial
    center = source.mean(axis=0)
    anchor = scale0 * rotation0 @ center + translation0
    centers_a = np.array([reference.cameras[f].pose[:3, 3] for f in training])
    centers_b = np.array([local.cameras[f].pose[:3, 3] for f in training])
    rotations_a = np.array([reference.cameras[f].pose[:3, :3] for f in training])
    rotations_b = np.array([local.cameras[f].pose[:3, :3] for f in training])

    def decode(parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        scale = scale0 * np.exp(parameters[0])
        rotation = Rotation.from_rotvec(parameters[1:4]).as_matrix() @ rotation0
        translation = anchor + scene_scale * parameters[4:] - scale * rotation @ center
        return float(scale), rotation, translation

    def blocks(parameters: np.ndarray) -> dict[str, np.ndarray]:
        scale, rotation, translation = decode(parameters)
        relative = rotations_a.transpose(0, 2, 1) @ rotation @ rotations_b
        return {
            "landmarks": (scale * source @ rotation.T + translation - target) / (0.05 * scene_scale),
            "camera_centers": (scale * centers_b @ rotation.T + translation - centers_a) / (0.05 * scene_scale),
            "camera_rotations": Rotation.from_matrix(relative).as_rotvec() / np.radians(10),
        }

    def residual(parameters: np.ndarray) -> np.ndarray:
        result = []
        for values in blocks(parameters).values():
            # Soft-L1 on each normalized vector norm, averaged per evidence
            # group. Hundreds of points must not drown out a few cameras.
            squared = np.sum(values * values, axis=1)
            weight = np.sqrt(2 / (np.sqrt(1 + squared) + 1))
            result.append((values * weight[:, None] / np.sqrt(len(values))).ravel())
        return np.concatenate(result)

    zero = np.zeros(7)
    report.update(training_camera_frames=training, held_out_camera_frames=held,
                  objective="Equal-weight group means: soft-L1 landmark, camera-center and camera-rotation residuals",
                  normalization={"position_scene_fraction": 0.05, "rotation_degrees": 10},
                  initial={"scale": scale0, "rotation": rotation0.tolist(), "translation": translation0.tolist()},
                  initial_objective=float(residual(zero) @ residual(zero)))
    lower, upper = np.full(7, -np.inf), np.full(7, np.inf)
    lower[0], upper[0] = -np.log(4), np.log(4)
    solution = least_squares(residual, zero, bounds=(lower, upper), method="trf",
                             max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    final_residual = residual(solution.x)
    report.update(converged=bool(solution.success), message=str(solution.message),
                  evaluations=int(solution.nfev), final_objective=float(final_residual @ final_residual),
                  scale_multiplier_bounds=[0.25, 4.0],
                  scale_bound_active=bool(solution.active_mask[0]),
                  training_groups={name: {"initial_rms": float(np.sqrt(np.mean(np.sum(values**2, axis=1)))),
                                         "final_rms": float(np.sqrt(np.mean(np.sum(blocks(solution.x)[name]**2, axis=1))))}
                                   for name, values in blocks(zero).items()})
    if (not solution.success or not np.isfinite(final_residual).all() or solution.active_mask[0]
            or report["final_objective"] > report["initial_objective"] + 1e-9):
        raise ValueError("Camera-aware Sim(3) optimizer did not produce a converged interior solution")
    report["status"] = "candidate_requires_validation"
    return decode(solution.x)
