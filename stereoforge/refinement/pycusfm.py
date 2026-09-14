"""Sequential monocular cuSFM adapter. Sparse refinement never relabels dense depth."""

from dataclasses import dataclass
import hashlib
import importlib.util
import importlib.metadata
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, tracked
from stereoforge.geometry.config import RefinementConfig
from stereoforge.geometry.types import GeometrySequence
from .tracks import NativeMatchesReader
from .landmarks import DepthLandmarkInitializer
from .colmap_validation import ColmapOutputValidator
from .colmap_binary import ColmapBinaryWriter, bundle_adjustment_command


def axis_angle(rotation: np.ndarray) -> dict:
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    if angle < 1e-8:
        axis = np.array([1., 0., 0.])
    elif math.pi - angle < 1e-5:
        _, vectors = np.linalg.eigh((rotation + rotation.T + 2 * np.eye(3)) / 4)
        axis = vectors[:, -1]
    else:
        axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]]) / (2 * math.sin(angle))
    return dict(zip(("x", "y", "z", "angle_degrees"), [*axis.tolist(), math.degrees(angle)], strict=True))


def pinhole_parameters(intrinsics: np.ndarray, height: int, width: int) -> dict:
    """Match upstream generate_frame_meta.py's explicit PINHOLE protobuf schema."""
    k = np.asarray(intrinsics, dtype=np.float64)
    if height <= 0 or width <= 0 or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Invalid pyCuSFM image dimensions or intrinsics")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("pyCuSFM requires positive focal lengths and homogeneous pinhole intrinsics")
    if not np.allclose([k[0, 1], k[1, 0]], 0, atol=1e-6, rtol=0):
        raise ValueError("Nonzero intrinsic skew is not supported by this pyCuSFM pinhole adapter")
    projection = np.column_stack((k, np.zeros(3)))
    return {"camera_projection_model_type": "PINHOLE", "calibration_parameters": {
        "image_width": width, "image_height": height,
        "projection_matrix": {"data": projection.reshape(-1).tolist(), "row_count": 3, "column_count": 4}}}


@dataclass(frozen=True, slots=True)
class CuSFMResult:
    directory: Path
    summary: dict


class CuSFMRefiner:
    STAGES = (("feature_extractor", "Feature extraction"), ("vocab_generator", "Vocabulary building"),
              ("pose_graph", "Pose graph optimization"), ("matcher", "Feature matching"),
              ("bundle_adjustment", "Depth-initialized bundle adjustment"))

    def __init__(self, config: RefinementConfig) -> None:
        self.config = config

    @property
    def feature_name(self) -> str:
        return "ALIKED"

    def preflight(self) -> Path:
        if shutil.which("cusfm_cli") is None:
            raise RuntimeError("cusfm_cli is unavailable; rebuild the pyCuSFM Docker image")
        spec = importlib.util.find_spec("pycusfm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Cannot locate the installed pycusfm package")
        package = Path(next(iter(spec.submodule_search_locations)))
        if not (package / "bin/bundle_adjustment_runner").is_file():
            raise RuntimeError("The installed pyCuSFM package lacks bundle_adjustment_runner; rebuild its Docker image")
        NativeMatchesReader()  # Check the protobuf dependency before expensive stages.
        model_directory = "aliked_lightglue"
        assets = ("aliked.onnx", "lightglue_aliked.onnx", "lightglue_aliked_b16.onnx")
        for name in assets:
            path = package / "models" / model_directory / name
            if not path.is_file() or path.stat().st_size < 1024:
                raise RuntimeError(f"Missing {self.feature_name}/LightGlue model asset: {path}; fetch upstream Git LFS assets and rebuild")
        return package

    def run(self, sequence: GeometrySequence, timestamps: tuple[float, ...] | None,
            directory: Path) -> CuSFMResult:
        package = self.preflight()
        if self.config.device >= torch.cuda.device_count():
            raise ValueError("refinement.device is outside the visible CUDA devices")
        directory.mkdir()
        inputs = directory / "input"
        inputs.mkdir()
        configs = directory / "config"
        shutil.copytree(package / "configs/isaac", configs)
        # Cache model engines in a writable directory, keyed by ONNX content so
        # an upstream model change cannot accidentally reuse an old engine.
        digest = hashlib.sha256()
        properties = torch.cuda.get_device_properties(self.config.device)
        digest.update(f"{properties.name}:{properties.major}.{properties.minor}".encode())
        digest.update(importlib.metadata.version("tensorrt-cu13").encode())
        model_source = package / "models/aliked_lightglue"
        for path in sorted(model_source.rglob("*.onnx")):
            digest.update(str(path.relative_to(package)).encode())
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "stereoforge/cusfm" / digest.hexdigest()[:16]
        models = cache / "models"
        shutil.copytree(model_source, models / "aliked_lightglue", dirs_exist_ok=True)
        export = self._export(sequence, timestamps, inputs)
        # An uncut forward-moving recording need not revisit any location.
        # Pose-graph loop associations alone can therefore yield no match tasks.
        # Select nearby cameras using the same working units as exported poses.
        pairs = configs / "match_pair_select_config.pb.txt"
        pair_text = pairs.read_text()
        pair_text = re.sub(r"(?m)^search_method:.*$", "search_method: RADIUS_SEARCH", pair_text)
        pair_text = re.sub(r"(?m)^search_translation_threshold_meters:.*$",
                           f"search_translation_threshold_meters: {export['matching_radius']}", pair_text)
        pair_text = re.sub(r"(?m)^min_time_interval_between_keyframes_seconds:.*$",
                           "min_time_interval_between_keyframes_seconds: 0", pair_text)
        pair_text = re.sub(r"(?m)^min_distance_between_keyframes_meters:.*$",
                           "min_distance_between_keyframes_meters: 0", pair_text)
        pair_text = re.sub(r"(?m)^use_downsampling:.*$", "use_downsampling: false", pair_text)
        pairs.write_text(pair_text)
        env = os.environ.copy()
        # Preserve the meaning of an index into the parent's visible GPU list.
        visible = env.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else None
        if devices is not None and self.config.device >= len(devices):
            raise ValueError("refinement.device is outside CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = devices[self.config.device] if devices else str(self.config.device)
        output = directory / "workspace"
        initialization = {}
        validation = {}
        base = ["cusfm_cli", "--input_dir", str(inputs), "--cusfm_base_dir", str(output),
                "--config_dir", str(configs), "--model_dir", str(models), "--feature_type", self.config.feature_type,
                "--skip_cuvslam", "true", "--ba_frame_type", "camera_frame",
                "--optimize_extrinsics", "false", "--optimize_intrinsics", "false",
                "--export_binary_colmap_files", "false",
                "--output_rgb", "true", "--min_inter_frame_distance", "0",
                "--min_inter_frame_rotation_degrees", "0", "--downsampling_matches", "false",
                "--stereo_pair_non_baseline_max_distance", "0"]
        if self.config.debug:
            base.extend(("--enable_debug", "true", "--debug_interval", "1"))
        for stage, label in self.STAGES:
            if stage == "feature_extractor":
                label = f"{self.feature_name} features"
            command = [*base, "--steps_to_run", stage]
            stage_env = env.copy()
            if stage == "bundle_adjustment":
                initialization = DepthLandmarkInitializer().build(sequence, directory, export)
                with Progress("Writing binary BA input"):
                    ColmapBinaryWriter().convert(directory / "initialized_sparse", directory / "initialized_binary")
                # Invoke the bundled binary directly: cusfm_tool splits an argument
                # string on whitespace and therefore cannot safely handle spaced paths.
                command = bundle_adjustment_command(package, directory)
                if env.get("USE_SYSTEM_PROTOBUF", "false").lower() != "true":
                    stage_env["LD_LIBRARY_PATH"] = str(package / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
            log = directory / f"{stage}.log"
            with Progress(f"pyCuSFM · {label}"), log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, env=stage_env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    status = process.wait()
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    raise
                if status:
                    stream.flush()
                    lines = log.read_text(errors="replace").splitlines()
                    # Keep the original assertion as well as the trailing stack trace.
                    relevant = [i for i, line in enumerate(lines)
                                if re.search(r"Check failed|FATAL|^F\d{4}|terminate called|what\(\)", line)]
                    excerpt = lines[max(0, relevant[0]-3):relevant[0]+10] if relevant else lines[:10]
                    diagnostic = directory.parent.parent / "failed_refinements" / uuid4().hex
                    try:
                        diagnostic.mkdir(parents=True)
                        shutil.copy2(log, diagnostic / log.name)
                        shutil.copy2(inputs / "frames_meta.json", diagnostic / "frames_meta.json")
                        saved = f"Full stage log and input metadata saved in {diagnostic}"
                    except OSError as exc:
                        saved = f"Could not preserve diagnostic files: {exc}"
                    detail = "\n".join([*excerpt, "...", *lines[-25:]])
                    raise RuntimeError(f"pyCuSFM {stage} failed (exit {status}):\n{detail}\n{saved}")
                if stage == "bundle_adjustment":
                    stream.flush()
                    mapper_text = log.read_text(errors="replace")
                    if "bundle adjustment failed" in mapper_text.lower() or "BA solve failed" in mapper_text:
                        raise RuntimeError(
                            "pyCuSFM reported a bundle-adjustment failure despite exiting successfully."
                            " Inspect bundle_adjustment.log; initialized_sparse retains the VGGT-seeded input."
                        )
                    if not list((output / "ba_raw").rglob("images.txt")):
                        raise RuntimeError("Bundle adjustment did not export COLMAP text output; inspect bundle_adjustment.log")
                    validation = ColmapOutputValidator().write_filtered(
                        output / "ba_raw", output / "sparse", directory / "initialized_sparse")
                if stage == "matcher":
                    tasks = list((output / "matches/tasks").glob("*.pb"))
                    matches = [p for p in (output / "matches").glob("*.pb")
                               if p.name != "map_keypoints.pb" and p.stat().st_size > 0]
                    if not tasks or not matches:
                        raise RuntimeError(
                            f"pyCuSFM matching produced {len(tasks)} task files and {len(matches)} match files. "
                            "Skipping depth initialization: inspect matcher.log and the pair-selection configuration."
                        )
                    # Native success and nonempty protobufs can still represent
                    # thousands of pairs with zero correspondences.
                    statistics = re.findall(
                        r"Number of frame pair:\s*(\d+),\s*Num Matches per frame pair:\s*([\d.eE+-]+)",
                        log.read_text(errors="replace"),
                    )
                    if statistics and all(float(average) == 0 for _, average in statistics):
                        pairs = sum(int(count) for count, _ in statistics)
                        raise RuntimeError(
                            f"{self.feature_name}/LightGlue attempted {pairs} pairs but reported zero matches per pair. "
                            "Stopping before bundle adjustment. Use the saved-VGGT refinement diagnostic "
                            "with --debug to inspect matching outputs; nonempty match files do not imply valid matches."
                        )
        summary = {**export, "feature_type": self.config.feature_type, "stages": [s for s, _ in self.STAGES],
                   "camera_to_camera_extrinsics": "not applicable: one physical camera; rig transform fixed",
                   "camera_pose_optimization": True, "intrinsics_optimization": "native standalone BA policy",
                   "initialization": initialization, "landmark_initialization": "vggt_depth",
                   "validation": validation,
                   "bundle_adjustment_backend": "bundle_adjustment_runner",
                   "input_format": "colmap_binary",
                   "dense_depth_refined": False, "pair_selection": "RADIUS_SEARCH"}
        write_json(directory / "refinement.json", summary)
        return CuSFMResult(directory, summary)

    @staticmethod
    def _export(sequence: GeometrySequence, timestamps: tuple[float, ...] | None, output: Path) -> dict:
        poses = np.stack([f.camera_to_world.numpy() for f in sequence.frames]).astype(np.float64)
        anchor = np.linalg.inv(poses[0])
        poses = anchor @ poses
        steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
        positive = steps[steps > 1e-8]
        if len(positive) == 0:
            raise ValueError("Monocular refinement requires camera translation/parallax")
        factor = 1.0 if sequence.units == "meters" else 1.0 / float(np.median(positive))
        poses[:, :3, 3] *= factor
        times = np.array(timestamps if timestamps is not None else np.arange(len(sequence)) / 30)
        micros = np.rint((times - times[0]) * 1e6).astype(np.int64)
        if len(micros) != len(sequence) or np.any(np.diff(micros) <= 0):
            raise ValueError("cuSFM needs distinct increasing microsecond timestamps")
        frequency = 1e6 / float(np.median(np.diff(micros))) if len(micros) > 1 else 30.
        frames, cameras = [], {}
        identity = {"axis_angle": {"x": 1, "y": 0, "z": 0, "angle_degrees": 0},
                    "translation": {"x": 0, "y": 0, "z": 0}}
        with tracked(sequence.frames, "Preparing pyCuSFM inputs", len(sequence), "frame") as pending:
            for index, frame in enumerate(pending):
                filename = f"frame_{index:06d}.png"
                pixels = sequence.processed_rgb[index].numpy().transpose(1, 2, 0)
                Image.fromarray((pixels * 255).round().astype(np.uint8)).save(output / filename, compress_level=1)
                h, w = frame.image_size_hw
                # Per-frame calibration IDs retain VGGT's per-frame K, while
                # sensor_id=0 identifies the single physical camera. No fake rig.
                cameras[str(index)] = {"sensor_meta_data": {"sensor_id": 0, "sensor_type": "CAMERA",
                    "sensor_name": "monocular", "frequency": frequency, "sensor_to_vehicle_transform": identity},
                    **pinhole_parameters(frame.intrinsics.numpy(), h, w)}
                frames.append({"id": str(index), "camera_params_id": str(index),
                    "timestamp_microseconds": str(micros[index]), "image_name": filename,
                    "synced_sample_id": str(index), "camera_to_world": {
                    "axis_angle": axis_angle(poses[index, :3, :3]),
                    "translation": dict(zip(("x", "y", "z"), poses[index, :3, 3].tolist(), strict=True))}})
        write_json(output / "frames_meta.json", {"keyframes_metadata": frames, "initial_pose_type": "EGO_MOTION",
                   "camera_params_id_to_camera_params": cameras,
                   "camera_params_id_to_session_name": {key: "0" for key in cameras}})
        return {"input_frames": len(sequence), "working_units": "meters" if sequence.units == "meters" else "normalized_monocular_units",
                "working_units_per_input_unit": factor,
                "matching_radius": max(1, math.ceil(float(np.median(positive)) * factor * 8)),
                "initial_world_to_first_camera": anchor.tolist(),
                "timestamp_source": "video" if timestamps is not None else "synthetic_30fps",
                "scale_note": "Uncalibrated runs use median camera step=1; metric-named upstream thresholds are working-unit priors, not measured meters"}
