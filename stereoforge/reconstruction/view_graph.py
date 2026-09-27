"""Verified temporal image graph, globally identified tracks and separator hierarchy."""

from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path

import numpy as np

from stereoforge.utils.progress import Progress

@dataclass(frozen=True, slots=True, order=True)
class Observation:
    frame: int
    keypoint: int


@dataclass(slots=True)
class Cluster:
    identifier: int
    frames: list[int]
    children: list["Cluster"] = field(default_factory=list)

    def document(self) -> dict:
        return {"id": self.identifier, "frames": self.frames,
                "children": [child.document() for child in self.children]}


class VerifiedGraph:
    def __init__(self, count: int, retain_feature_tracks: bool = False) -> None:
        self.count = count
        self.retain_feature_tracks = retain_feature_tracks
        self.neighbors: dict[int, set[int]] = {i: set() for i in range(count)}
        self.pixels: dict[Observation, np.ndarray] = {}
        self.tracks: list[dict[int, np.ndarray]] = []
        self.summary: dict = {}
        self.observation_tracks: dict[Observation, int] = {}

    def read(self, files: list[Path], on_progress=None, on_tracks=None, on_orbit=None,
             description: str = "Building measured feature tracks") -> None:
        if on_progress is not None:
            # Stage events clear Rerun's transient matching links. Send this
            # before orbit updates, which intentionally preserve active tracks.
            on_progress(f"{description}: reading verified matches")
        with Progress(description, sum(path.stat().st_size for path in files), "B") as progress:
            self._read(files, progress, on_progress, on_tracks, on_orbit)

    def _read(self, files: list[Path], progress: Progress, on_progress=None, on_tracks=None, on_orbit=None) -> None:
        progress.status("reading verified matches")
        edges = []
        total_bytes = max(1, sum(path.stat().st_size for path in files))
        bytes_read = 0
        if on_orbit is not None:
            on_orbit(0.0)
        verified_pairs = 0
        for path in files:
            with path.open('rb') as stream:
                for line in stream:
                    progress.advance(len(line))
                    bytes_read += len(line)
                    if on_orbit is not None:
                        on_orbit(0.3 * bytes_read / total_bytes)
                    pair = json.loads(line)
                    a, b = pair["source"], pair["target"]
                    if a not in self.neighbors or b not in self.neighbors or a == b:
                        raise ValueError("Matcher returned invalid global frame IDs")
                    matches = [m for m in pair["matches"] if m["inlier"]]
                    if len(matches) < 30 or len(matches) < 0.25 * len(pair["matches"]):
                        continue
                    verified_pairs += 1
                    self.neighbors[a].add(b)
                    self.neighbors[b].add(a)
                    for match in matches:
                        nodes = []
                        for frame, feature_key, pixel_key in ((a, "source_feature", "a"), (b, "target_feature", "b")):
                            if feature_key not in match:
                                raise ValueError("Pair matcher lacks stable feature IDs; rebuild the Docker image")
                            node = Observation(frame, int(match[feature_key]))
                            uv = np.asarray(match[pixel_key], dtype=np.float64)
                            if uv.shape != (2,) or not np.isfinite(uv).all():
                                raise ValueError("Invalid matched feature coordinates")
                            if node in self.pixels and not np.allclose(self.pixels[node], uv, atol=1e-4, rtol=0):
                                raise ValueError("Feature identity changed between pairs; refusing inconsistent tracks")
                            self.pixels[node] = uv
                            nodes.append(node)
                        confidence = float(match["confidence"])
                        if not np.isfinite(confidence):
                            raise ValueError("Invalid feature confidence")
                        edges.append((confidence, *nodes))
        # Merge strongest edges first. Reject only conflicting edges, not the
        # entire existing component and its otherwise valid observations.
        parent: dict[Observation, Observation] = {}
        members: dict[Observation, dict[int, Observation]] = {}

        def root(node: Observation) -> Observation:
            if node not in parent:
                parent[node] = node
                members[node] = {node.frame: node}
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        if on_progress is not None:
            on_progress(f"Building tracks: sorting {len(edges):,} verified match edges")
        progress.status(f"sorting {len(edges):,} match edges")
        edges.sort(reverse=True)
        progress.reset(len(edges), "edge", "joining consistent tracks")
        rejected = 0
        track_samples = []
        joined = 0

        def show_track(group, label):
            if on_tracks is None:
                return
            nodes = sorted(group.values())[:8]
            identity = f"{nodes[0].frame}:{nodes[0].keypoint}"
            track_samples.append((identity, {node.frame: self.pixels[node] for node in nodes}))
            del track_samples[:-4]
            on_tracks(label, track_samples)

        for edge_index, (_, a, b) in enumerate(edges):
            if on_orbit is not None and edge_index % 10000 == 0:
                on_orbit(0.3 + 0.5 * edge_index / max(1, len(edges)))
            x, y = root(a), root(b)
            if x == y:
                progress.advance()
                continue
            if members[x].keys() & members[y].keys():
                rejected += 1
                progress.advance()
                continue
            if len(members[x]) < len(members[y]):
                x, y = y, x
            parent[y] = x
            members[x].update(members.pop(y))
            joined += 1
            if joined == 1 or joined % 20000 == 0:
                show_track(members[x], f"Joining measured tracks: {edge_index+1:,}/{len(edges):,} edges — schematic pixel links")
            progress.advance()
        if on_progress is not None:
            on_progress(f"Building tracks: {len(members):,} components; collecting tracks with at least three views")
        progress.reset(len(members), "track", "collecting tracks with at least three views")
        self.tracks = []
        for group_index, group in enumerate(members.values()):
            if on_orbit is not None and group_index % 5000 == 0:
                on_orbit(0.8 + 0.15 * group_index / max(1, len(members)))
            if len(group) >= 3:
                if self.retain_feature_tracks:
                    self.observation_tracks.update({node: len(self.tracks) for node in group.values()})
                self.tracks.append({f: self.pixels[node] for f, node in sorted(group.items())})
                if len(self.tracks) == 1 or len(self.tracks) % 20000 == 0:
                    show_track(group, f"{len(self.tracks):,} tracks built · {group_index+1:,}/{len(members):,} components processed")
            progress.advance()
        progress.reset(self.count, "frame", "checking image connectivity")
        components, pending = [], set(self.neighbors)
        while pending:
            component, frontier = set(), [min(pending)]
            while frontier:
                frame = frontier.pop()
                if frame not in component:
                    component.add(frame)
                    progress.advance()
                    frontier.extend(self.neighbors[frame] - component)
            components.append(sorted(component))
            pending -= component
        if on_orbit is not None:
            on_orbit(1.0)
        self.summary = {"verified_pairs": verified_pairs, "match_edges": len(edges),
                        "conflicting_edges_rejected": rejected, "tracks": len(self.tracks),
                        "components": components, "frontend": "RaCo-ALIKED/LightGlue+; native F/H RANSAC"}

    def partition(self, maximum: int = 32, overlap: int = 8) -> Cluster:
        if len(self.summary["components"]) != 1:
            raise ValueError("Verified image graph is disconnected; inspect graph.json. No frames were silently dropped")
        counter = 0

        def connected(frames: set[int]) -> bool:
            seen, frontier = set(), [min(frames)]
            while frontier:
                frame = frontier.pop()
                if frame not in seen:
                    seen.add(frame)
                    frontier.extend((self.neighbors[frame] & frames) - seen)
            return seen == frames

        def divide(frames: set[int]) -> Cluster:
            nonlocal counter
            node = Cluster(counter, sorted(frames))
            counter += 1
            if len(frames) <= maximum:
                return node
            # BFS graph order, independent of estimated camera positions.
            order, seen, queue = [], set(), deque([min(frames)])
            while queue:
                frame = queue.popleft()
                if frame in seen:
                    continue
                seen.add(frame)
                order.append(frame)
                queue.extend(sorted((self.neighbors[frame] & frames) - seen))
            if len(order) != len(frames):
                raise ValueError("A proposed cluster is disconnected")
            candidates = []
            for cut in range(max(1, len(order)*2//5), min(len(order), len(order)*3//5+1)):
                left, right = set(order[:cut]), set(order[cut:])
                left_boundary = {f for f in left if self.neighbors[f] & right}
                right_boundary = {f for f in right if self.neighbors[f] & left}
                separator = min((left_boundary, right_boundary), key=lambda s: (len(s), sorted(s))).copy()
                boundary = left_boundary | right_boundary
                pool = set(boundary)
                while len(pool) < overlap:
                    expanded = pool | {n for f in pool for n in self.neighbors[f] & frames}
                    if expanded == pool:
                        break
                    pool = expanded
                for f in sorted(pool, key=lambda f: (f not in boundary, -len(self.neighbors[f] & frames), f)):
                    if len(separator) >= overlap:
                        break
                    separator.add(f)
                a, b = left | separator, right | separator
                if (len(separator) >= 6 and max(len(a), len(b)) < len(frames)
                        and connected(a) and connected(b)):
                    candidates.append((len(separator), max(len(a), len(b)), sorted(a), sorted(b)))
            if not candidates:
                raise ValueError("Cannot partition the verified graph with sufficient shared cameras; inspect hierarchy inputs")
            _, _, a, b = min(candidates)
            node.children = [divide(set(a)), divide(set(b))]
            return node

        return divide(set(self.neighbors))
