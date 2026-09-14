"""Read the pinned cuSFM wire format and assemble unambiguous image tracks.

Field numbers/types were read from the protobuf descriptors embedded in the
bundled CUDA-13 feature_matcher_main. Only required fields are declared; protobuf
preserves compatibility with unused fields. No descriptor bytes are guessed from
individual result files. Revisit this schema when upgrading the native binaries.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereoforge.utils.progress import tracked


@dataclass(frozen=True, slots=True, order=True)
class Observation:
    frame: int
    keypoint: int


class NativeMatchesReader:
    def __init__(self) -> None:
        try:
            from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        except ImportError as exc:
            raise RuntimeError("Native match import requires protobuf: uv pip install 'protobuf>=5,<7'") from exc
        schema = descriptor_pb2.FileDescriptorProto(name="stereoforge_cusfm_tracks.proto",
                                                    package="stereoforge_native", syntax="proto3")
        # Protobuf types: float=2, uint64=4, int32=5, message=11, uint32=13.
        definitions = {
            "KeypointVector": [("x", 1, 2, True, None), ("y", 2, 2, True, None),
                               ("width", 6, 5, False, None), ("height", 7, 5, False, None)],
            "Keyframe": [("points", 6, 11, False, "KeypointVector"), ("id", 8, 4, False, None)],
            "FramePair": [("source", 1, 4, False, None), ("target", 2, 4, False, None)],
            "FeatureMatch": [("source", 1, 13, False, None), ("target", 2, 13, False, None)],
            "FramePairMatch": [("frames", 1, 11, False, "FramePair"),
                               ("matches", 2, 11, True, "FeatureMatch")],
            "FrameMatches": [("pairs", 1, 11, True, "FramePairMatch")],
        }
        for name, fields in definitions.items():
            message = schema.message_type.add(name=name)
            for field_name, number, kind, repeated, target in fields:
                field = message.field.add(name=field_name, number=number, type=kind,
                                          label=3 if repeated else 1)
                if target:
                    field.type_name = f".stereoforge_native.{target}"
        pool = descriptor_pool.DescriptorPool()
        pool.Add(schema)
        self._keyframe = message_factory.GetMessageClass(pool.FindMessageTypeByName("stereoforge_native.Keyframe"))
        self._matches = message_factory.GetMessageClass(pool.FindMessageTypeByName("stereoforge_native.FrameMatches"))

    def keypoints(self, directory: Path, image_sizes: list[tuple[int, int]]) -> dict[int, np.ndarray]:
        result = {}
        files = sorted(directory.glob("frame_*.pb"))
        with tracked(files, "Reading ALIKED keypoints", len(files), "frame") as pending:
            for path in pending:
                record = self._keyframe.FromString(path.read_bytes())
                index = int(path.stem.removeprefix("frame_"))
                # Current native files omit frame_id: identity is provided by
                # the exported filename/frames_meta.json. Check it if populated.
                if (record.id and record.id != index) or index >= len(image_sizes) or index in result:
                    raise ValueError(f"Unexpected native frame ID in {path}")
                height, width = image_sizes[index]
                points = record.points
                if (points.height, points.width) != (height, width) or len(points.x) != len(points.y):
                    raise ValueError(f"Native keypoint schema/image dimensions disagree in {path}")
                uv = np.column_stack((points.x, points.y)).astype(np.float64)
                if not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv >= [width, height]):
                    raise ValueError(f"Native keypoints are outside processed pixel coordinates in {path}")
                result[index] = uv
        if not result:
            raise ValueError("No ALIKED keypoint files were found")
        return result

    def tracks(self, directory: Path, keypoints: dict[int, np.ndarray]) -> tuple[list[list[Observation]], dict]:
        forest = TrackForest()
        files = sorted(directory.glob("matching_task_*_result.pb"))
        edges = 0
        with tracked(files, "Building ALIKED tracks", len(files), "file") as pending:
            for path in pending:
                record = self._matches.FromString(path.read_bytes())
                for pair in record.pairs:
                    source, target = pair.frames.source, pair.frames.target
                    if source == target or source not in keypoints or target not in keypoints:
                        raise ValueError(f"Invalid matched frame pair in {path}")
                    for match in pair.matches:
                        if match.source >= len(keypoints[source]) or match.target >= len(keypoints[target]):
                            raise ValueError(f"Match references an unknown keypoint in {path}")
                        forest.join(Observation(source, match.source), Observation(target, match.target))
                        edges += 1
        tracks, conflicts = forest.tracks()
        if not edges:
            raise ValueError("Native matching supplied no observations for depth initialization")
        return tracks, {"match_edges": edges, "candidate_tracks": len(tracks),
                        "ambiguous_tracks_rejected": conflicts}


class TrackForest:
    """Union-find; reject entire components containing competing observations."""

    def __init__(self) -> None:
        self._parent: dict[Observation, Observation] = {}
        self._size: dict[Observation, int] = {}

    def _root(self, node: Observation) -> Observation:
        if node not in self._parent:
            self._parent[node] = node
            self._size[node] = 1
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while node != root:
            parent = self._parent[node]
            self._parent[node] = root
            node = parent
        return root

    def join(self, first: Observation, second: Observation) -> None:
        first, second = self._root(first), self._root(second)
        if first == second:
            return
        if self._size[first] < self._size[second]:
            first, second = second, first
        self._parent[second] = first
        self._size[first] += self._size[second]

    def tracks(self) -> tuple[list[list[Observation]], int]:
        groups: dict[Observation, list[Observation]] = {}
        for node in self._parent:
            groups.setdefault(self._root(node), []).append(node)
        tracks, conflicts = [], 0
        for group in groups.values():
            if len({node.frame for node in group}) != len(group):
                conflicts += 1
            elif len(group) >= 3:
                tracks.append(sorted(group))
        return sorted(tracks, key=lambda track: track[0]), conflicts
