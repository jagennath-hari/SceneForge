"""Import COLMAP text output and visualize sparse refinement separately from depth."""

from copy import deepcopy
import html
import json
from pathlib import Path
import re

import numpy as np

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.visualization import GeometryReportWriter


def write_refinement_status(directory: Path, message: str, status: str = "failed") -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"status": status, "message": message}
    write_json(directory / "status.json", summary)
    links = "".join(f'<li><a href="{html.escape(p.name, quote=True)}">{html.escape(p.name)}</a></li>'
                    for p in sorted(directory.glob("*.log")))
    (directory / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>pyCuSFM diagnostics</title>'
        '<style>body{background:#101620;color:#eee;font:16px system-ui;margin:40px}'
        'a{color:#79cafa}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>pyCuSFM refinement</h1><p><a href="../index.html">View unrefined VGGT geometry</a></p>'
        f'<p>Status: {html.escape(status)}</p><pre>{html.escape(message)}</pre>'
        f'<h2>Stage logs</h2><ul>{links}</ul>', encoding="utf-8")
    return summary


def rotation_from_quaternion(values: list[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = quaternion / norm
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def write_refinement_report(directory: Path, original: dict, max_points: int) -> dict:
    candidates = list((directory / "workspace/sparse").rglob("images.txt"))
    if len(candidates) != 1:
        raise ValueError("Expected one connected COLMAP text reconstruction from pyCuSFM")
    folder = candidates[0].parent
    cameras = {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        camera_id, model, width, height = parts[:4]
        params = list(map(float, parts[4:]))
        if model == "SIMPLE_PINHOLE":
            fx, cx, cy = params
            fy = fx
        elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
            fx, fy, cx, cy = params[:4]
            if any(abs(v) > 1e-8 for v in params[4:]):
                raise ValueError("Refined camera has distortion; the viewer needs an explicit undistortion adapter")
        else:
            raise ValueError(f"Unsupported refined camera model: {model}")
        cameras[int(camera_id)] = ([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], [int(height), int(width)])
    registered = []
    with candidates[0].open() as file:
        while line := file.readline():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 10:
                raise ValueError("Malformed COLMAP image record")
            match = re.fullmatch(r"frame_(\d+)\.png", Path(parts[9]).name)
            if match is None:
                raise ValueError(f"Cannot match refined frame: {parts[9]}")
            index = int(match[1])
            if index >= len(original["frames"]):
                raise ValueError("Unknown frame index in refined reconstruction")
            r = rotation_from_quaternion(list(map(float, parts[1:5])))
            t = np.array(list(map(float, parts[5:8])))
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = r.T, -r.T @ t
            registered.append((index, int(parts[0]), int(parts[8]), pose))
            # Second line contains 2D observations and can legitimately be empty.
            if file.readline() == "":
                raise ValueError("Truncated COLMAP image observations")
    registered.sort(key=lambda item: item[0])
    if len({i for i, _, _, _ in registered}) != len(registered):
        raise ValueError("Duplicate frame IDs in the refined reconstruction; inspect workspace/sparse/images.txt")
    if not registered:
        return write_refinement_status(directory, "pyCuSFM exported zero registered cameras. "
                                       "No after-reconstruction is available; the VGGT report and native outputs are retained.", "no_reconstruction")
    refined = np.stack([p for _, _, _, p in registered])
    reference = np.array([original["frames"][i]["camera_to_world"] for i, _, _, _ in registered])
    aligned = False
    alignment_note = "Too few registered cameras for a reliable comparison; showing native cuSFM coordinates."
    rotation, translation, scale = np.eye(3), np.zeros(3), 1.0
    source, target = refined[:, :3, 3].copy(), reference[:, :3, 3]
    if len(registered) >= 3:
        try:
            # Camera orientations resolve rotation even for a nearly straight flight path.
            u, _, vt = np.linalg.svd(np.sum(reference[:, :3, :3] @ refined[:, :3, :3].transpose(0, 2, 1), axis=0))
            signs = np.diag([1., 1., -1. if np.linalg.det(u @ vt) < 0 else 1.])
            rotation = u @ signs @ vt
            source, target = refined[:, :3, 3] @ rotation.T, reference[:, :3, 3]
            a, b = source.mean(axis=0), target.mean(axis=0)
            variance = np.sum((source - a) ** 2)
            if variance < 1e-12:
                raise ValueError("Refined trajectory has insufficient translation for scale alignment")
            scale = float(np.sum((source-a)*(target-b)) / variance)
            translation = b - scale*a
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid refined-to-VGGT scale alignment")
            aligned = True
            alignment_note = "Aligned to VGGT for comparison; local drift is preserved."
        except (ValueError, np.linalg.LinAlgError) as exc:
            rotation, translation, scale = np.eye(3), np.zeros(3), 1.0
            alignment_note = f"Showing native cuSFM coordinates: {exc}"
    frames, id_to_order = [], {}
    for order, (index, image_id, camera_id, pose) in enumerate(registered):
        record = deepcopy(original["frames"][index])
        pose[:3, :3] = rotation @ pose[:3, :3]
        pose[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
        record["camera_to_world"] = pose.tolist()
        record["intrinsics"], record["processed_size_hw"] = cameras[camera_id]
        record["previews"] = {kind: "../" + name for kind, name in record["previews"].items()}
        frames.append(record)
        id_to_order[image_id] = order
    # Reservoir sampling bounds viewer memory without biasing toward early points.
    rng = np.random.default_rng(0)
    points, colors, point_frames, errors = [], [], [], []
    count, error_sum = 0, 0.
    with (folder / "points3D.txt").open() as point_file:
        for line in point_file:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            xyz = np.array(list(map(float, parts[1:4])))
            color = list(map(int, parts[4:7]))
            error = float(parts[7])
            track = [id_to_order[int(image)] for image in parts[8::2] if int(image) in id_to_order]
            if not track or not np.isfinite(xyz).all() or not np.isfinite(error):
                continue
            xyz = scale * (rotation @ xyz) + translation
            count += 1
            error_sum += error
            slot = count - 1 if count <= max_points else int(rng.integers(count))
            if slot >= max_points:
                continue
            if count <= max_points:
                points.append(xyz); colors.append(color); point_frames.append(min(track)); errors.append(error)
            else:
                points[slot], colors[slot], point_frames[slot], errors[slot] = xyz, color, min(track), error

    summary = {"registered_frames": len(frames), "input_frames": len(original["frames"]),
               "registered_fraction": len(frames)/len(original["frames"]), "sparse_points": count,
               "mean_point_reprojection_error_pixels": error_sum/count if count else None,
               "missing_frame_indices": sorted(set(range(len(original["frames"]))) - {i for i, _, _, _ in registered}),
               "alignment_to_vggt": {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()} if aligned else None,
               "alignment_note": alignment_note,
               "status": "complete" if aligned and count and len(frames) == len(original["frames"]) else "partial",
               "trajectory_rms_after_alignment": float(np.sqrt(np.mean(np.sum((scale*source+translation-target)**2, axis=1)))) if aligned else None,
               "note": "Alignment removes global gauge differences for comparison, not local drift. Dense depth remains the original VGGT estimate."}
    metadata = deepcopy(original)
    refinement_path = directory / "refinement.json"
    refinement = json.loads(refinement_path.read_text()) if refinement_path.is_file() else {}
    feature_type = refinement.get("feature_type")
    feature_name = "ALIKED" if feature_type == "aliked" else None
    reconstruction_name = f"pyCuSFM · {feature_name}" if feature_name else "pyCuSFM"
    summary["feature_type"] = feature_type
    metadata.update(frames=frames, reconstruction_name=reconstruction_name, sparse_refinement=True,
                    refinement_link="../index.html", refinement_label="View VGGT reconstruction", refinement_summary=summary)
    if not aligned:
        metadata["units"] = "reconstruction_units"
        metadata["meters_per_unit"] = None
    times = original.get("provenance", {}).get("video_timestamps_seconds")
    if times:
        metadata["provenance"]["video_timestamps_seconds"] = [times[i] for i, _, _, _ in registered]
    write_json(directory / "comparison.json", summary)
    write_json(directory / "metadata.json", metadata)
    xyz, rgb = np.asarray(points).reshape(-1, 3), np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    GeometryReportWriter._write_ply(directory / "point_cloud.ply", xyz, rgb)
    GeometryReportWriter._write_viewer(directory, metadata, xyz, rgb, np.asarray(point_frames, dtype=np.int64))
    return summary
