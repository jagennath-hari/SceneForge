"""Graph-first hierarchical sparse reconstruction using the exposed pyCuSFM BA."""

from dataclasses import dataclass
import filecmp
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from stereoforge.geometry.adaptive import _load
from .registration import align_sparse_sections, merge_sparse_sections
from stereoforge.refinement.bundle_adjustment import SparseBundleAdjuster, model_statistics
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel
from stereoforge.geometry.config import RefinementConfig
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group, tracked
from stereoforge.utils.visualization import GeometryReportWriter
from stereoforge.video.sampling import VideoFrameSampler
from .inference import infer_clusters
from .merge_validation import MergeValidation, POLICY, model_fingerprint
from .triangulation import TrackTriangulator
from .view_graph import Cluster, VerifiedGraph


@dataclass(frozen=True, slots=True)
class HierarchyOptions:
    cluster_size: int = 32
    overlap: int = 8
    neighbors: int = 4
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not 16 <= self.cluster_size <= 64:
            raise ValueError("cluster_size must be between 16 and 64")
        if not 6 <= self.overlap <= self.cluster_size // 2:
            raise ValueError("overlap must be between 6 and half the cluster size")
        if not 1 <= self.neighbors <= 12:
            raise ValueError("neighbors must be between 1 and 12")
        if self.device != "cuda" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("Use cuda for all visible GPUs, or cuda:N")


class HierarchicalReconstructor:
    def __init__(self, options: HierarchyOptions, output: Path, keyframe_config: Path) -> None:
        self.options, self.output, self.keyframe_config = options, output, keyframe_config
        digest = hashlib.sha256()
        refinement = Path(__file__).resolve().parents[1] / "refinement"
        sources = [*Path(__file__).parent.glob("*.py"), refinement / "bundle_adjustment.py",
                   refinement / "sparse_model.py", refinement / "colmap_validation.py",
                   refinement / "colmap_binary.py",
                   Path(__file__).resolve().parents[1] / "geometry/alignment.py"]
        for source in sorted(sources):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        self.merge_implementation = digest.hexdigest()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the hierarchical frontend and pyCuSFM")
        self.devices = ([f"cuda:{i}" for i in range(torch.cuda.device_count())] if options.device == "cuda"
                        else [options.device])
        self.device = int(self.devices[0].split(":")[1])
        if self.device >= torch.cuda.device_count():
            raise ValueError("Requested device is outside the visible CUDA devices")
        self.optimizer = SparseBundleAdjuster(RefinementConfig(device=self.device))
        self.package = self.optimizer.preflight()
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the hierarchical pair matcher")
        capabilities = subprocess.run([executable, "--capabilities"], capture_output=True, text=True)
        if capabilities.returncode or not json.loads(capabilities.stdout).get("stable_feature_ids"):
            raise RuntimeError("Pair matcher needs rebuilding: missing stable feature ID support")

    def run(self, paths: list[Path], checkpoint: Path, timestamps: tuple[float, ...] | None = None) -> dict:
        if len(paths) < 6:
            raise ValueError("Hierarchical reconstruction requires at least six selected keyframes")
        self.timestamps = timestamps
        self.images = self._prepare_images(paths)
        files = self._match(len(paths))
        graph = VerifiedGraph(len(paths))
        with Progress("Building verified image graph and tracks"):
            graph.read(files)
        write_json(self.output / "graph.json", graph.summary)
        hierarchy = graph.partition(self.options.cluster_size, self.options.overlap)
        hierarchy_path = self.output / "hierarchy.json"
        if hierarchy_path.exists() and json.loads(hierarchy_path.read_text()) != hierarchy.document():
            raise ValueError("Saved hierarchy differs; start a new run to avoid reusing unrelated clusters")
        write_json(hierarchy_path, hierarchy.document())
        # Persist the shared track identity, independently of any cluster geometry.
        track_path = self.output / "tracks.jsonl"
        temporary_tracks = self.output / "tracks.partial"
        with temporary_tracks.open("w") as stream:
            for identifier, track in enumerate(graph.tracks):
                stream.write(json.dumps({"id": identifier, "observations": {str(f): uv.tolist() for f, uv in track.items()}}) + "\n")
        if track_path.exists():
            if not filecmp.cmp(track_path, temporary_tracks, shallow=False):
                raise ValueError("Saved global tracks differ; start a new run")
        temporary_tracks.replace(track_path)
        leaves = []

        def collect(node: Cluster) -> None:
            if not node.children:
                leaves.append(node)
            for child in node.children:
                collect(child)

        collect(hierarchy)
        with Progress("Preparing small VGGT clusters") as progress:
            progress.status(f"{len(leaves)} groups; at most {self.options.cluster_size} images per group")
        infer_clusters(checkpoint, paths, leaves, self.output / "vggt", self.devices)
        # All inference workers have exited before any BA process starts.
        triangulator = TrackTriangulator(graph.tracks, self.images)
        (self.output / "nodes").mkdir(exist_ok=True)

        def reconstruct(node: Cluster) -> SparseModel:
            folder = self.output / "nodes" / str(node.identifier)
            folder.mkdir(exist_ok=True)
            completion = folder / "ba/complete.json"
            validation = None
            dependencies = None
            if not node.children:
                if completion.is_file():
                    return SparseModel.read(folder / "ba/workspace/sparse", node.frames)
                sequence = _load(self.output / "vggt" / f"{node.identifier}.pt")
                if [f.frame_index for f in sequence.frames] != node.frames:
                    raise ValueError("Saved VGGT cluster identity does not match the hierarchy")
                cameras = {}
                for i, frame in enumerate(sequence.frames):
                    with Image.open(self.images[frame.frame_index]) as image:
                        expected = np.array(image)
                    actual = (sequence.processed_rgb[i].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                    if not np.array_equal(actual, expected):
                        raise ValueError("VGGT and global feature tracks use different processed RGB grids")
                    cameras[frame.frame_index] = SparseCamera(frame.camera_to_world.numpy().astype(float),
                                                              frame.intrinsics.numpy().astype(float), frame.image_size_hw)
                del sequence
                seed, details = triangulator.build(cameras)
                write_json(folder / "triangulation.json", details)
            else:
                reference = reconstruct(node.children[0])
                local = reconstruct(node.children[1])
                dependencies = [model_fingerprint(reference), model_fingerprint(local)]
                if completion.is_file():
                    saved = json.loads(completion.read_text())
                    if (saved.get("policy") == POLICY and saved.get("dependencies") == dependencies
                            and saved.get("implementation") == self.merge_implementation):
                        return SparseModel.read(folder / "ba/workspace/sparse", node.frames)
                alignment = {"status": "rejected"}
                try:
                    aligned = align_sparse_sections(reference, local, alignment)
                finally:
                    write_json(folder / "alignment.json", alignment)
                details = {}
                try:
                    seed = merge_sparse_sections(reference, aligned, details)
                finally:
                    write_json(folder / "merge.json", details)
                validation = MergeValidation.prepare(seed, reference, aligned)
                write_json(folder / "withheld_observations.json", validation.document())
                write_json(folder / "validation_before_ba.json", validation.evaluate(seed))
                seed = validation.training
            if len(seed.cameras) < 6 or len(seed.points) < 30:
                raise ValueError(f"Cluster {node.identifier} has insufficient sparse support; inspect {folder}")
            write_json(folder / "seed_summary.json", {**model_statistics(seed),
                       "missing_input_frames": sorted(set(node.frames)-seed.cameras.keys())})
            return self._ba(seed, folder / "ba", node.frames, validation, dependencies)

        with progress_group("Hierarchical reconstruction"):
            merged = reconstruct(hierarchy)
        final_directory = self.output / "final"
        root_fingerprint = [model_fingerprint(merged)]
        final_marker = final_directory / "complete.json"
        saved_final = json.loads(final_marker.read_text()) if final_marker.is_file() else {}
        if (saved_final.get("dependencies") == root_fingerprint and saved_final.get("policy") == POLICY
                and saved_final.get("implementation") == self.merge_implementation):
            final = SparseModel.read(final_directory / "workspace/sparse", list(range(len(paths))))
        else:
            # Rebuild from the ORIGINAL global observations, not only surviving
            # local landmark coordinates or previously truncated merge tracks.
            seed, details = triangulator.build(merged.cameras)
            write_json(self.output / "global_retriangulation.json", details)
            final = self._ba(seed, final_directory, list(range(len(paths))), dependencies=root_fingerprint)
        missing = sorted(set(range(len(paths)))-final.cameras.keys())
        summary = {"status": "partial" if missing else "complete", "input_frames": len(paths),
                   "missing_frames": missing, "clusters": len(leaves), **model_statistics(final),
                   "optimizer": "pyCuSFM standalone BA", "frontend": "RaCo-ALIKED/LightGlue+",
                   "partitioner": "verified-graph BFS balanced separators; not METIS nested dissection",
                   "priors": "No explicit GTSAM pose/calibration priors or GNC-TLS; native policy, Cauchy loss",
                   "final_ba": "Global retriangulation followed by the same native BA; no prior-release switch exposed",
                   "units": "reconstruction_units", "dense_depth_refined": False}
        write_json(self.output / "report.json", summary)
        self._viewer(final, summary, timestamps)
        return summary

    def _ba(self, seed: SparseModel, directory: Path, ids: list[int],
            validation: MergeValidation | None = None, dependencies: list[str] | None = None) -> SparseModel:
        if directory.exists():
            # Preserve previous attempts; never mix native outputs with a retry.
            directory.rename(directory.with_name(directory.name + ".previous_" + uuid4().hex[:8]))
        result = self.optimizer.optimize_sparse(seed, directory, ids, self.package)
        report = None
        if validation is not None:
            report = validation.evaluate(result)
            write_json(directory.parent / "validation_after_ba.json", report)
        self._viewer(result, {"input_frames": len(ids), **model_statistics(result),
                     "status": report["status"] if report else "optimized",
                     "merge_validation": report}, self.timestamps, directory)
        if report is not None and report["status"] != "accepted":
            raise ValueError(f"Joint BA failed withheld-overlap validation ({report['within_5px']:.1%} within 5px); "
                             f"inspect {directory.parent / 'validation_after_ba.json'}")
        write_json(directory / "complete.json", {"frames": sorted(result.cameras), "points": len(result.points),
                   "policy": POLICY, "dependencies": dependencies, "implementation": self.merge_implementation})
        return result

    def _prepare_images(self, paths: list[Path]) -> dict[int, Path]:
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        folder = self.output / "processed"
        folder.mkdir(exist_ok=True)
        result = {}
        sizes = set()
        with tracked(enumerate(paths), "Preparing feature images", len(paths), "frame") as pending:
            for index, path in pending:
                destination = folder / f"{index:06d}.png"
                if not destination.exists():
                    tensor = load_and_preprocess_images([str(path)], mode="balanced", image_resolution=512)[0]
                    pixels = (tensor.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                    temporary = destination.with_suffix(".partial")
                    Image.fromarray(pixels).save(temporary, format="PNG", compress_level=1)
                    temporary.rename(destination)
                with Image.open(destination) as image:
                    sizes.add(image.size)
                result[index] = destination
        if len(sizes) != 1:
            raise ValueError("Hierarchical video input must use a uniform processed image size")
        self.size_wh = next(iter(sizes))
        return result

    def _match(self, count: int) -> list[Path]:
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the pair matcher with global feature IDs")
        folder = self.output / "matching"
        folder.mkdir(exist_ok=True)
        models = None
        result = []
        width, height = self.size_wh
        batches = range(0, max(count - 1, 0), 64)
        if any(not (folder / f"{start:06d}.jsonl").is_file() for start in batches):
            models = VideoFrameSampler.model_arguments(self.keyframe_config)[1]
        with Progress("Verifying temporal image pairs", total=len(batches), unit="batch") as progress:
            for start in batches:
                destination = folder / f"{start:06d}.jsonl"
                if destination.is_file():
                    result.append(destination)
                    progress.status(f"reused batch · source frames {start}–{min(start+64, count-1)-1}")
                    progress.advance()
                    continue
                pairs = [(a, b) for a in range(start, min(start+64, count))
                         for b in range(a+1, min(a+self.options.neighbors+1, count))]
                if not pairs:
                    continue
                request = folder / f"{start:06d}.request.json"
                write_json(request, {"pairs": [{"source": a, "target": b, "width": width, "height": height,
                             "source_path": str(self.images[a]), "target_path": str(self.images[b])} for a, b in pairs]})
                temporary = destination.with_suffix(".partial_" + uuid4().hex[:8])
                command = [executable, str(request), str(self.keyframe_config), models, str(temporary)]
                if self.device:
                    command.append(str(self.device))
                progress.status(f"source frames {start}–{min(start+64, count-1)-1}")
                with destination.with_suffix(".log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Verify identity and schema before marking this batch reusable.
                records = [json.loads(line) for line in temporary.read_text().splitlines()]
                if [(r["source"], r["target"]) for r in records] != pairs:
                    raise ValueError("Pair matcher returned incomplete or unexpected frame IDs")
                if any("source_feature" not in m for r in records for m in r["matches"]):
                    raise ValueError("Rebuild Docker: pair matcher lacks stable global feature IDs")
                temporary.rename(destination)
                result.append(destination)
                progress.advance()
        return result

    def _viewer(self, model: SparseModel, summary: dict, timestamps: tuple | None,
                directory: Path | None = None) -> None:
        destination = directory if directory is not None else self.output
        frames, order = [], {f: i for i, f in enumerate(sorted(model.cameras))}
        for frame, camera in sorted(model.cameras.items()):
            frames.append({"frame_index": frame, "source": f"Keyframe {frame}",
                           "processed_size_hw": camera.size_hw, "camera_to_world": camera.pose.tolist(),
                           "intrinsics": camera.intrinsics.tolist(), "valid_fraction": 0, "depth_p50": None,
                           "previews": {"rgb": os.path.relpath(self.images[frame], destination)}})
        metadata = {"format_version": 1, "units": "reconstruction_units", "meters_per_unit": None,
                    "reconstruction_name": "Hierarchical pyCuSFM", "sparse_only": True,
                    "frames": frames, "input_frames": summary["input_frames"], "sparse_statistics": summary,
                    "dense_depth_refined": False, "provenance": {"video_timestamps_seconds":
                        [timestamps[f] for f in sorted(model.cameras)] if timestamps else None}}
        write_json(destination / "metadata.json", metadata)
        selected = np.linspace(0, len(model.points)-1, min(240000, len(model.points)), dtype=int)
        points = [model.points[i] for i in selected]
        xyz, rgb = np.array([p.xyz for p in points]), np.array([p.rgb for p in points], dtype=np.uint8)
        appeared = np.array([min(order[f] for f in p.observations) for p in points], dtype=np.int64)
        GeometryReportWriter._write_ply(destination / "point_cloud.ply", xyz, rgb)
        GeometryReportWriter._write_viewer(destination, metadata, xyz, rgb, appeared)
