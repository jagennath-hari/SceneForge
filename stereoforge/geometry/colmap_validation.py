"""Validate standalone BA output and retain measured, reprojection-consistent tracks."""

from pathlib import Path

import numpy as np

from .cusfm_report import rotation_from_quaternion


class ColmapOutputValidator:
    def __init__(self, max_error: float = 3.0, min_observations: int = 3) -> None:
        self.max_error = max_error
        self.min_observations = min_observations

    def write_filtered(self, source: Path, destination: Path, initialized: Path) -> dict:
        candidates = list(source.rglob("images.txt"))
        if len(candidates) != 1:
            raise ValueError("Expected one COLMAP text reconstruction from standalone BA")
        folder = candidates[0].parent
        cameras = {}
        camera_lines = {}
        for line in (folder / "cameras.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            identifier = int(fields[0])
            if identifier in cameras or fields[1] != "PINHOLE" or len(fields) != 8:
                raise ValueError("Standalone BA must export unique PINHOLE cameras for this viewer")
            width, height = int(fields[2]), int(fields[3])
            fx, fy, cx, cy = map(float, fields[4:])
            if min(width, height, fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
                raise ValueError("Invalid optimized camera intrinsics")
            cameras[identifier] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            camera_lines[identifier] = line
        # Only names actually supplied to BA are allowed back into the result.
        expected_names = set()
        with (initialized / "images.txt").open() as stream:
            while line := stream.readline():
                if line.strip() and not line.startswith("#"):
                    expected_names.add(line.split()[-1])
                    stream.readline()
        images = {}
        with candidates[0].open() as stream:
            while line := stream.readline():
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != 10:
                    raise ValueError("Malformed optimized COLMAP image")
                identifier, camera_id = int(fields[0]), int(fields[8])
                if identifier in images or camera_id not in cameras or fields[9] not in expected_names:
                    raise ValueError("Unknown or duplicate optimized image/camera")
                rotation = rotation_from_quaternion(list(map(float, fields[1:5])))
                translation = np.array(list(map(float, fields[5:8])))
                if not np.isfinite(translation).all():
                    raise ValueError("Invalid optimized camera translation")
                observation_line = stream.readline()
                if observation_line == "":
                    raise ValueError("Truncated optimized observations")
                values = observation_line.split()
                if len(values) % 3:
                    raise ValueError("Malformed optimized observations")
                observations = [(float(values[i]), float(values[i+1]), int(values[i+2]))
                                for i in range(0, len(values), 3)]
                images[identifier] = (line.strip(), camera_id, rotation, translation, observations)
        points = []
        seen = set()
        claimed_observations = set()
        input_points = 0
        with (folder / "points3D.txt").open() as stream:
            for line in stream:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 8 or (len(fields)-8) % 2:
                    raise ValueError("Malformed optimized landmark")
                identifier = int(fields[0])
                if identifier in seen:
                    raise ValueError("Duplicate optimized landmark ID")
                seen.add(identifier)
                input_points += 1
                xyz = np.array(list(map(float, fields[1:4])))
                if not np.isfinite(xyz).all():
                    continue
                observations, errors, frames = [], [], set()
                for j in range(8, len(fields), 2):
                    image_id, index = int(fields[j]), int(fields[j+1])
                    if image_id not in images or index < 0 or index >= len(images[image_id][4]):
                        raise ValueError("Optimized landmark refers to a missing observation")
                    if image_id in frames or (image_id, index) in claimed_observations:
                        raise ValueError("Ambiguous optimized landmark observations")
                    frames.add(image_id)
                    claimed_observations.add((image_id, index))
                    _, camera_id, rotation, translation, uv = images[image_id]
                    u, v, point_id = uv[index]
                    if point_id != identifier:
                        raise ValueError("COLMAP image and landmark associations disagree")
                    camera = rotation @ xyz + translation
                    if camera[2] <= 0 or not np.isfinite([u, v, *camera]).all():
                        continue
                    pixel = cameras[camera_id] @ camera
                    error = float(np.linalg.norm(pixel[:2]/pixel[2] - [u, v]))
                    if error <= self.max_error:
                        observations.append((image_id, index))
                        errors.append(error)
                if len(observations) >= self.min_observations:
                    points.append((identifier, fields[1:7], observations, float(np.mean(errors))))
        if not points:
            raise ValueError("No landmarks passed post-BA reprojection validation; raw BA output is retained in workspace/ba_raw")
        retained_images: dict[int, list[tuple[int, int]]] = {}
        tracks = {}
        for identifier, _, observations, _ in points:
            tracks[identifier] = []
            for image_id, old_index in observations:
                records = retained_images.setdefault(image_id, [])
                tracks[identifier].append((image_id, len(records)))
                records.append((old_index, identifier))
        destination.mkdir()
        with (destination / "images.txt").open("w") as stream:
            for image_id, observations in sorted(retained_images.items()):
                stream.write(images[image_id][0] + "\n")
                stream.write(" ".join(f"{images[image_id][4][i][0]:.17g} {images[image_id][4][i][1]:.17g} {pid}"
                                      for i, pid in observations) + "\n")
        used_cameras = {images[i][1] for i in retained_images}
        (destination / "cameras.txt").write_text("\n".join(camera_lines[i] for i in sorted(used_cameras)) + "\n")
        with (destination / "points3D.txt").open("w") as stream:
            for identifier, geometry, _, error in points:
                stream.write(f"{identifier} " + " ".join(geometry) + f" {error:.17g} "
                             + " ".join(f"{i} {j}" for i, j in tracks[identifier]) + "\n")
        return {"native_landmarks": input_points, "validated_landmarks": len(points),
                "validated_frames": len(retained_images), "post_ba_max_reprojection_error": self.max_error,
                "min_track_observations": self.min_observations}
