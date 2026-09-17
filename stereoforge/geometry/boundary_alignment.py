"""Register adjacent sections exclusively through their identical shared frames."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch

from .alignment import OverlapAlignmentError, SimilarityTransform, _points, align_overlap
from .geometry_utils import confidence_filtered_mask
from .types import FrameGeometry, GeometrySequence


def align_boundary(
    reference: dict[int, FrameGeometry], local: GeometrySequence,
    reference_rgb: Mapping[int, torch.Tensor], *, icp_device: str = "auto",
) -> SimilarityTransform:
    """Fit on shared training images; validate on separate shared images.

    No image outside the intersection can seed registration or ICP. Failure
    requests boundary recovery; it does not trigger a distant-frame search or
    silently publish a disconnected map. The transform maps all local geometry
    into the reference world, but does not repair internal local distortion.
    """
    shared = [frame for frame in local.frames if frame.frame_index in reference]
    details = {"policy": "shared_frames_only", "status": "boundary_recovery_required",
               "shared_frame_indices": [frame.frame_index for frame in shared],
               "direction": "local_to_reference", "rgb_checks": []}
    positions = {frame.frame_index: i for i, frame in enumerate(local.frames)}
    usable = []
    for frame in shared:
        index = frame.frame_index
        identical = index in reference_rgb and torch.equal(
            reference_rgb[index], local.processed_rgb[positions[index]])
        details["rgb_checks"].append({"frame_index": index, "identical": identical})
        if not identical:
            details["status"] = "input_mismatch"
            raise OverlapAlignmentError(f"Shared frame {index} RGB differs; check inputs/preprocessing", details)
        if frame.image_size_hw != reference[index].image_size_hw:
            details["status"] = "input_mismatch"
            raise OverlapAlignmentError(f"Shared frame {index} geometry grids differ", details)
        mask_a, _ = confidence_filtered_mask(frame.depth, frame.confidence, 50)
        mask_b, _ = confidence_filtered_mask(reference[index].depth, reference[index].confidence, 50)
        if int((mask_a & mask_b).sum()) >= 32:
            usable.append(frame)
    details["usable_frame_indices"] = [frame.frame_index for frame in usable]
    if len(usable) < 5:
        raise OverlapAlignmentError(
            "Boundary recovery required: need at least five shared frames with reliable depth "
            "(three for fitting and two for validation); no outside frames were searched", details)
    # Split whole images before any transform/ICP estimation. Interleave them
    # across the shared interval so validation is not limited to one endpoint.
    validation = usable[::3]
    held_ids = {frame.frame_index for frame in validation}
    training = [frame for frame in usable if frame.frame_index not in held_ids]
    details["training_frame_indices"] = [frame.frame_index for frame in training]
    details["validation_frame_indices"] = sorted(held_ids)
    selected = [positions[frame.frame_index] for frame in training]
    subset = GeometrySequence(tuple(training), local.processed_rgb[selected],
                              tuple(local.source_names[i] for i in selected),
                              tuple(local.original_sizes_hw[i] for i in selected),
                              local.units, local.meters_per_unit)
    try:
        transform = align_overlap(
            {frame.frame_index: reference[frame.frame_index] for frame in training},
            subset, {frame.frame_index: reference_rgb[frame.frame_index] for frame in training},
            icp_device=icp_device)
    except OverlapAlignmentError as exc:
        details["training"] = exc.diagnostics
        raise OverlapAlignmentError(f"Boundary recovery required: {exc}. No outside frames were searched", details) from exc
    details["training"] = transform.as_dict()
    checks = []
    for frame in validation:
        other = reference[frame.frame_index]
        mask_a, _ = confidence_filtered_mask(frame.depth, frame.confidence, 50)
        mask_b, _ = confidence_filtered_mask(other.depth, other.confidence, 50)
        y, x = torch.where(mask_a & mask_b)
        selected_pixels = torch.linspace(0, len(y)-1, min(512, len(y))).long()
        y, x = y[selected_pixels], x[selected_pixels]
        source, target = _points(frame, y, x), _points(other, y, x)
        moved = transform.scale * (source @ transform.rotation.T) + transform.translation
        errors = np.linalg.norm(moved-target, axis=1) / other.depth[y, x].double().numpy()
        delta = (transform.rotation @ frame.camera_to_world[:3, :3].double().numpy()
                 @ other.camera_to_world[:3, :3].double().numpy().T)
        angle = float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))))
        median, fraction = float(np.median(errors)), float(np.mean(errors <= .08))
        checks.append({"frame_index": frame.frame_index, "relative_error": median,
                       "inlier_fraction": fraction, "pose_error_degrees": angle,
                       "passed": median <= .08 and fraction >= .5 and angle <= 10})
    details["validation"] = checks
    details["thresholds"] = {"median_relative_error": .08, "minimum_inlier_fraction": .5,
                              "maximum_pose_error_degrees": 10, "every_validation_frame_must_pass": True}
    if not all(item["passed"] for item in checks):
        raise OverlapAlignmentError(
            "Boundary recovery required: held-out shared frames disagree with the fitted transform; "
            "no outside frames were searched", details)
    details["status"] = "accepted"
    details["scope"] = "One transform for the entire local section; internal errors outside overlap are not validated"
    return replace(transform, shared_frames=len(usable), diagnostics=details)
