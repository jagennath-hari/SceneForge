"""Register sparse BA sections using observations in identical shared images."""

from dataclasses import replace

import numpy as np

from stereoforge.geometry.alignment import _fit
from stereoforge.refinement.sparse_model import SparseModel, SparsePoint, reprojection_error


def shared_landmarks(reference: SparseModel, local: SparseModel) -> list[tuple[int, int]]:
    """Conservative identity: same image and ALIKED coordinate to 0.0001 pixel.

    No nearest-surface matching or distant-frame search. Ambiguous landmark
    associations are excluded rather than joined transitively.
    """
    shared = reference.cameras.keys() & local.cameras.keys()
    lookup: dict[tuple[int, float, float], set[int]] = {}
    for i, point in enumerate(reference.points):
        for frame, uv in point.observations.items():
            if frame in shared:
                lookup.setdefault((frame, *np.round(uv, 4)), set()).add(i)
    forward: dict[int, set[int]] = {}
    backward: dict[int, set[int]] = {}
    for j, point in enumerate(local.points):
        for frame, uv in point.observations.items():
            for i in lookup.get((frame, *np.round(uv, 4)), ()):
                forward.setdefault(i, set()).add(j)
                backward.setdefault(j, set()).add(i)
    return [(i, next(iter(matches))) for i, matches in sorted(forward.items())
            if len(matches) == 1 and len(backward[next(iter(matches))]) == 1]


def align_sparse_sections(reference: SparseModel, local: SparseModel, report: dict) -> SparseModel:
    pairs = shared_landmarks(reference, local)
    shared = sorted(reference.cameras.keys() & local.cameras.keys())
    report.update(shared_registered_frames=shared, shared_landmarks=len(pairs),
                  coordinate_tolerance_pixels=0.0001)
    if len(shared) < 6 or len(pairs) < 60:
        raise ValueError("Local BA left insufficient shared support: need 6 cameras and 60 unambiguous landmarks")
    source = np.array([local.points[j].xyz for _, j in pairs])
    target = np.array([reference.points[i].xyz for i, _ in pairs])
    # Normalize the geometric gate by scene depth, independent of either BA gauge.
    distances = []
    for i, _ in pairs:
        point = reference.points[i]
        distances.extend(np.linalg.norm(point.xyz - reference.cameras[f].pose[:3, 3])
                         for f in point.observations if f in shared)
    scene_scale = float(np.median(distances))
    if not np.isfinite(scene_scale) or scene_scale <= 1e-8:
        raise ValueError("Degenerate overlap scene scale")
    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(pairs))
    held = permutation[::3]
    train = np.setdiff1d(permutation, held)
    threshold = 0.05 * scene_scale
    best, best_mask = None, np.zeros(len(train), dtype=bool)
    for _ in range(512):
        sample = rng.choice(train, 3, replace=False)
        try:
            candidate = _fit(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        scale, rotation, translation = candidate
        errors = np.linalg.norm(scale * source[train] @ rotation.T + translation - target[train], axis=1)
        mask = errors <= threshold
        if mask.sum() > best_mask.sum():
            best, best_mask = candidate, mask
    if best is None or best_mask.sum() < 20:
        raise ValueError("No robust Sim(3) from locally refined shared landmarks")
    scale, rotation, translation = _fit(source[train[best_mask]], target[train[best_mask]])
    errors = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
    aligned = local.transformed(scale, rotation, translation)
    camera_checks = []
    for frame in shared:
        a, b = reference.cameras[frame], aligned.cameras[frame]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(a.pose[:3, :3].T @ b.pose[:3, :3])-1)/2, -1, 1))))
        distance = float(np.linalg.norm(a.pose[:3, 3] - b.pose[:3, 3]) / scene_scale)
        camera_checks.append({"frame": frame, "rotation_degrees": angle,
                              "relative_translation_error": distance,
                              "passed": angle <= 10 and distance <= 0.05})
    # Independent landmarks were not used for fitting; check measured pixels too.
    pixel_errors = []
    for p in held:
        i, j = pairs[p]
        point = aligned.points[j]
        for frame, uv in reference.points[i].observations.items():
            if frame in shared:
                pixel_errors.append(reprojection_error(reference.cameras[frame], point.xyz, uv))
        for frame, uv in local.points[j].observations.items():
            if frame in shared:
                pixel_errors.append(reprojection_error(aligned.cameras[frame], reference.points[i].xyz, uv))
    pixel_fraction = float(np.mean(np.asarray(pixel_errors) <= 5)) if pixel_errors else 0.0
    report.update(scale=scale, rotation=rotation.tolist(), translation=translation.tolist(),
                  scene_scale=scene_scale, training_landmarks=len(train), held_out_landmarks=len(held),
                  train_inlier_fraction=float(np.mean(errors[train] <= threshold)),
                  held_out_inlier_fraction=float(np.mean(errors[held] <= threshold)),
                  held_out_reprojection_within_5px=pixel_fraction, cameras=camera_checks)
    if (report["train_inlier_fraction"] < 0.8 or report["held_out_inlier_fraction"] < 0.8
            or not all(c["passed"] for c in camera_checks)):
        raise ValueError("Refined sections still disagree: sparse alignment validation failed; inspect alignment.json")
    report["status"] = "provisional"
    report["acceptance_requires"] = "Joint BA followed by withheld-observation validation"
    return aligned


def merge_sparse_sections(reference: SparseModel, local: SparseModel, report: dict) -> SparseModel:
    """One camera per global frame; fuse only unambiguous measured tracks."""
    pairs = shared_landmarks(reference, local)
    partners = dict(pairs)
    paired_local = {j for _, j in pairs}
    cameras = {**local.cameras, **reference.cameras}
    candidates: list[tuple[SparsePoint, bool]] = []
    conflicts = 0
    for i, point in enumerate(reference.points):
        if i not in partners:
            candidates.append((point, False))
            continue
        other = local.points[partners[i]]
        common = point.observations.keys() & other.observations.keys()
        if any(np.linalg.norm(point.observations[f] - other.observations[f]) > 1e-3 for f in common):
            conflicts += 1
            continue
        observations = {**other.observations, **point.observations}
        # Choose the seed with lower median error under the reconciled cameras.
        xyz = min((point.xyz, other.xyz), key=lambda value: np.median([
            reprojection_error(cameras[f], value, uv) for f, uv in observations.items()]))
        candidates.append((replace(point, xyz=xyz, observations=observations), True))
    candidates.extend((p, False) for j, p in enumerate(local.points) if j not in paired_local)
    points = []
    bridges = 0
    joined_count = 0
    reference_only = reference.cameras.keys() - local.cameras.keys()
    local_only = local.cameras.keys() - reference.cameras.keys()
    claimed = set()
    for point, joined in candidates:
        observations = {f: uv for f, uv in point.observations.items()
                        if reprojection_error(cameras[f], point.xyz, uv) <= 14}
        keys = {(f, *np.round(uv, 4)) for f, uv in observations.items()}
        if len(observations) < 3 or keys & claimed:
            continue
        claimed.update(keys)
        points.append(replace(point, observations=observations))
        joined_count += int(joined)
        # A bridge must really connect the sections, not just duplicate overlap tracks.
        if joined and observations.keys() & reference_only and observations.keys() & local_only:
            bridges += 1
    # Shared cameras themselves connect the optimization; joined overlap tracks
    # are still required even if no track spans the entire overlap interval.
    report.update(joined_track_candidates=len(pairs), conflicting_tracks=conflicts,
                  retained_landmarks=len(points), joined_tracks_passing_reprojection=joined_count,
                  tracks_spanning_both_nonoverlap_regions=bridges)
    if joined_count < 20:
        raise ValueError("Too few joined tracks survived reconciliation for combined BA")
    supported = {f for p in points for f in p.observations}
    missing = sorted(cameras.keys() - supported)
    report["lost_registered_frames"] = missing
    if missing:
        raise ValueError("Reconciliation removed all observations from registered cameras")
    return SparseModel(cameras, tuple(points))
