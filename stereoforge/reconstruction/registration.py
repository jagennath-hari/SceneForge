"""Register sparse BA sections using observations in identical shared images."""

from dataclasses import replace

import numpy as np

from .similarity import fit_similarity
from stereoforge.refinement.sparse_model import SparseModel, SparsePoint, reprojection_error
from .camera_alignment import refine_camera_alignment


def landmark_quality(model: SparseModel, point: SparsePoint) -> dict:
    """Depth observability from measured rays in one child, before alignment."""
    rays, errors = [], []
    for frame, uv in point.observations.items():
        camera = model.cameras[frame]
        ray = camera.pose[:3, :3] @ np.linalg.solve(camera.intrinsics, [*uv, 1])
        rays.append(ray / np.linalg.norm(ray))
        errors.append(reprojection_error(camera, point.xyz, uv))
    if len(rays) < 3 or not np.isfinite(rays).all() or not np.isfinite(errors).all():
        return {"reliable": False, "observations": len(rays), "angle_degrees": None}
    rays = np.asarray(rays)
    # Bound temporary storage for long tracks.
    cosine = min(float(np.min(rays[start:start+128] @ rays.T)) for start in range(0, len(rays), 128))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    median = float(np.median(errors))
    return {"reliable": bool(5 <= angle <= 90 and median <= 1.5 and max(errors) <= 3),
            "observations": len(rays), "angle_degrees": angle, "median_reprojection_pixels": median}


def landmark_coverage(model: SparseModel, indices: list[int], shared: list[int]) -> dict:
    frames = []
    for frame in shared:
        h, w = model.cameras[frame].size_hw
        pixels = [model.points[i].observations[frame] for i in indices if frame in model.points[i].observations]
        cells = {(min(3, max(0, int(4*uv[0]/w))), min(3, max(0, int(4*uv[1]/h)))) for uv in pixels}
        frames.append({"frame": frame, "observations": len(pixels), "grid_cells": len(cells),
                       "passed": len(pixels) >= 3 and len(cells) >= 3})
    return {"passed": sum(f["passed"] for f in frames) >= 6, "frames": frames,
            "grid": [4, 4], "minimum_supported_cameras": 6}


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
    qualities = [{"reference": landmark_quality(reference, reference.points[i]),
                  "local": landmark_quality(local, local.points[j])} for i, j in pairs]
    eligible = [p for p, q in enumerate(qualities) if q["reference"]["reliable"] and q["local"]["reliable"]]
    report.update(reliable_landmarks=len(eligible), weak_landmarks=len(pairs)-len(eligible),
                  reliability_policy={"minimum_angle_degrees": 5, "minimum_observations": 3,
                                      "maximum_median_reprojection_pixels": 1.5,
                                      "maximum_reprojection_pixels": 3},
                  landmark_quality=[{"reference_point": i, "local_point": j, **q}
                                    for (i, j), q in zip(pairs, qualities, strict=True)])
    if len(eligible) < 60:
        raise ValueError(f"Insufficient depth-observable shared landmarks: {len(eligible)}/60 required")
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
    # Split only after independent child-quality selection, never by cross-map residual.
    permutation = rng.permutation(eligible)
    held = permutation[::3]
    train = np.setdiff1d(permutation, held)
    coverage = {}
    for name, selection in (("training", train), ("held_out", held)):
        coverage[name] = {"reference": landmark_coverage(reference, [pairs[p][0] for p in selection], shared),
                          "local": landmark_coverage(local, [pairs[p][1] for p in selection], shared)}
    report["reliable_landmark_coverage"] = coverage
    if not all(side["passed"] for split in coverage.values() for side in split.values()):
        raise ValueError("Reliable alignment landmarks lack camera/image coverage in training or held-out set")
    threshold = 0.05 * scene_scale
    best, best_mask = None, np.zeros(len(train), dtype=bool)
    for _ in range(512):
        sample = rng.choice(train, 3, replace=False)
        try:
            candidate = fit_similarity(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        scale, rotation, translation = candidate
        errors = np.linalg.norm(scale * source[train] @ rotation.T + translation - target[train], axis=1)
        mask = errors <= threshold
        if mask.sum() > best_mask.sum():
            best, best_mask = candidate, mask
    if best is None or best_mask.sum() < 20:
        raise ValueError("No robust Sim(3) from locally refined shared landmarks")
    scale, rotation, translation = fit_similarity(source[train[best_mask]], target[train[best_mask]])
    camera_fit = {}
    report["camera_aware_refinement"] = camera_fit
    scale, rotation, translation = refine_camera_alignment(
        (scale, rotation, translation), source[train], target[train],
        reference, local, shared, scene_scale, camera_fit)
    errors = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
    aligned = local.transformed(scale, rotation, translation)
    camera_checks = []
    for frame in shared:
        a, b = reference.cameras[frame], aligned.cameras[frame]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(a.pose[:3, :3].T @ b.pose[:3, :3])-1)/2, -1, 1))))
        distance = float(np.linalg.norm(a.pose[:3, 3] - b.pose[:3, 3]) / scene_scale)
        camera_checks.append({"frame": frame, "rotation_degrees": angle,
                              "relative_translation_error": distance,
                              "used_for_alignment_fit": frame in camera_fit["training_camera_frames"],
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
    report["all_landmark_inlier_fraction_diagnostic"] = float(np.mean(errors <= threshold))
    failures = []
    for key, label in (("train_inlier_fraction", "training landmark agreement"),
                       ("held_out_inlier_fraction", "held-out landmark agreement")):
        if report[key] < 0.8:
            failures.append(f"{label} {report[key]:.1%} < 80%")
    failed_cameras = [c["frame"] for c in camera_checks if not c["passed"]]
    if failed_cameras:
        failures.append(f"shared-camera checks failed for frames {failed_cameras}")
    report["failed_checks"] = failures
    if failures:
        raise ValueError("Sparse alignment initialization rejected: " + "; ".join(failures))
    report["status"] = "provisional"
    report["acceptance_requires"] = "Joint BA followed by withheld-observation validation"
    return aligned


def merge_sparse_sections(reference: SparseModel, local: SparseModel, report: dict, *,
                          track_pairs: list[tuple[int, int]] | None = None) -> SparseModel:
    """One camera per global frame; fuse only unambiguous measured tracks."""
    pairs = shared_landmarks(reference, local) if track_pairs is None else track_pairs
    if len({i for i, _ in pairs}) != len(pairs) or len({j for _, j in pairs}) != len(pairs):
        raise ValueError('Landmark fusion requires one-to-one track identities')
    partners = dict(pairs)
    paired_local = {j for _, j in pairs}
    cameras = {**local.cameras, **reference.cameras}
    points: list[SparsePoint] = []
    claimed: set[tuple] = set()
    joined_count = bridges = conflicts = 0
    proposed = accepted = unchanged = 0
    reference_only = reference.cameras.keys() - local.cameras.keys()
    local_only = local.cameras.keys() - reference.cameras.keys()

    # The reference cameras remain unchanged during reconciliation. Its existing
    # points and observations are therefore valid seeds regardless of whether
    # an incoming track can extend them. Only joint BA may refine these seeds.
    for i, point in enumerate(reference.points):
        observations = dict(point.observations)
        added = 0
        if i in partners:
            other = local.points[partners[i]]
            common = point.observations.keys() & other.observations.keys()
            conflict = any(np.linalg.norm(point.observations[f] - other.observations[f]) > 1e-3
                           for f in common)
            additions = {f: uv for f, uv in other.observations.items() if f not in observations}
            proposed += len(additions)
            if conflict:
                conflicts += 1
            else:
                for frame, uv in additions.items():
                    if reprojection_error(cameras[frame], point.xyz, uv) <= 4:
                        observations[frame] = uv
                        added += 1
                # Count only compatible measured support from the incoming
                # window, never a fallback with no usable incoming observations.
                incoming_support = sum(
                    reprojection_error(cameras[f], point.xyz, uv) <= 4
                    for f, uv in other.observations.items())
                joined_count += int(incoming_support >= 3)
                if (incoming_support >= 3 and observations.keys() & reference_only
                        and observations.keys() & local_only):
                    bridges += 1
            accepted += added
            unchanged += int(added == 0)
        keys = {(f, *np.round(uv, 4)) for f, uv in observations.items()}
        if len(observations) < 3 or keys & claimed:
            raise ValueError('Established map track has insufficient or conflicting observations')
        claimed.update(keys)
        points.append(replace(point, observations=observations) if added else point)

    # New landmarks still need support under the reconciled cameras. An incoming
    # copy of an established global track is never inserted as a second point.
    for j, point in enumerate(local.points):
        if j in paired_local:
            continue
        observations = {f: uv for f, uv in point.observations.items()
                        if reprojection_error(cameras[f], point.xyz, uv) <= 14}
        keys = {(f, *np.round(uv, 4)) for f, uv in observations.items()}
        if len(observations) < 3 or keys & claimed:
            continue
        claimed.update(keys)
        points.append(replace(point, observations=observations))
    # Shared cameras themselves connect the optimization; joined overlap tracks
    # are still required even if no track spans the entire overlap interval.
    report.update(joined_track_candidates=len(pairs), conflicting_tracks=conflicts,
                  preserved_reference_landmarks=len(reference.points),
                  shared_tracks_kept_without_extension=unchanged,
                  proposed_observations=proposed, accepted_observations=accepted,
                  rejected_observations=proposed-accepted,
                  extension_max_reprojection_pixels=4,
                  reconciliation_policy="Preserve established tracks; gate incoming observations before joint BA",
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
