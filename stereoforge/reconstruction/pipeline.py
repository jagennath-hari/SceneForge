"""Graph-first hierarchical sparse reconstruction using the exposed pyCuSFM BA."""

from dataclasses import dataclass
import filecmp
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from stereoforge.geometry.adaptive import _load
from .registration import align_sparse_sections, merge_sparse_sections
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel
from stereoforge.refinement.bundle_adjustment import model_statistics
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group
from .frontend import ReconstructionFrontend
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


class HierarchicalReconstructor(ReconstructionFrontend):
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
                except ValueError as exc:
                    raise ValueError(f"Merge node {node.identifier}: {exc}. "
                                     f"Inspect {folder / 'alignment.json'}") from exc
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
