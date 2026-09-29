# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jagennath Hari
#
# This file is part of SceneForge.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verified temporal/loop image graph and globally identified feature tracks."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from sceneforge.utils.progress import Progress

@dataclass(frozen=True, slots=True, order=True)
class Observation:
    frame: int
    keypoint: int


@dataclass(slots=True)
class Cluster:
    identifier: int
    frames: list[int]

    def document(self) -> dict:
        return {"id": self.identifier, "frames": self.frames}


class VerifiedGraph:
    def __init__(self, count: int) -> None:
        self.count = count
        self.neighbors: dict[int, set[int]] = {i: set() for i in range(count)}
        self.pixels: dict[Observation, np.ndarray] = {}
        self.tracks: list[dict[int, np.ndarray]] = []
        self.summary: dict = {}
        self.temporal_crossing_counts = np.zeros(count, dtype=np.int64)
        self.loop_links: dict[int, set[tuple[int, int]]] = {}

    def read(self, files: list[Path], on_progress=None, on_tracks=None, on_orbit=None,
             description: str = "Building measured feature tracks",
             verified_loops: set[tuple[int, int]] | None = None) -> None:
        if on_progress is not None:
            # Stage events clear Rerun's transient matching links. Send this
            # before orbit updates, which intentionally preserve active tracks.
            on_progress(f"{description}: reading verified matches")
        with Progress(description, sum(path.stat().st_size for path in files), "B") as progress:
            self._read(files, progress, verified_loops or set(), on_progress, on_tracks, on_orbit)

    def _read(self, files: list[Path], progress: Progress, verified_loops: set[tuple[int, int]],
              on_progress=None, on_tracks=None, on_orbit=None) -> None:
        progress.status("reading verified matches")
        edges = []
        total_bytes = max(1, sum(path.stat().st_size for path in files))
        bytes_read = 0
        if on_orbit is not None:
            on_orbit(0.0)
        verified_pairs = 0
        loop_candidates = loop_geometry = corroborated_loops = 0
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
                    is_loop = pair.get('pair_kind', 'temporal') == 'loop'
                    if is_loop:
                        loop_candidates += 1
                        loop_geometry += bool(pair.get('loop_geometry', {}).get('accepted', False))
                        if (a, b) not in verified_loops:
                            continue
                    matches = [m for m in pair["matches"] if m["inlier"]]
                    if len(matches) < 30 or len(matches) < 0.25 * len(pair["matches"]):
                        continue
                    verified_pairs += 1
                    corroborated_loops += is_loop
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
                        edges.append((confidence, *nodes, is_loop))
        # Merge strongest edges first. Reject only conflicting edges, not the
        # entire existing component and its otherwise valid observations.
        parent: dict[Observation, Observation] = {}
        members: dict[Observation, dict[int, Observation]] = {}

        temporal_parent: dict[Observation, Observation] = {}
        temporal_members: dict[Observation, dict[int, Observation]] = {}
        loop_edges: list[tuple[Observation, Observation]] = []

        def root(node: Observation, parent=parent, members=members) -> Observation:
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

        for edge_index, (_, a, b, is_loop) in enumerate(edges):
            # A separate temporal forest prevents loops from hiding a weak local
            # boundary. Both forests share the same parsed matches and ordering.
            if not is_loop:
                tx = root(a, temporal_parent, temporal_members)
                ty = root(b, temporal_parent, temporal_members)
                if tx != ty and not (temporal_members[tx].keys() & temporal_members[ty].keys()):
                    if len(temporal_members[tx]) < len(temporal_members[ty]):
                        tx, ty = ty, tx
                    temporal_parent[ty] = tx
                    temporal_members[tx].update(temporal_members.pop(ty))
            if on_orbit is not None and edge_index % 10000 == 0:
                on_orbit(0.3 + 0.5 * edge_index / max(1, len(edges)))
            x, y = root(a), root(b)
            if x == y:
                if is_loop:
                    loop_edges.append((a, b))
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
            if is_loop:
                loop_edges.append((a, b))
            if joined == 1 or joined % 20000 == 0:
                show_track(members[x], f"Joining measured tracks: {edge_index+1:,}/{len(edges):,} edges — schematic pixel links")
            progress.advance()
        if on_progress is not None:
            on_progress(f"Building tracks: {len(members):,} components; collecting tracks with at least three views")
        progress.reset(len(members), "track", "collecting tracks with at least three views")
        self.tracks = []
        track_indices = {}
        for group_index, group in enumerate(members.values()):
            if on_orbit is not None and group_index % 5000 == 0:
                on_orbit(0.8 + 0.15 * group_index / max(1, len(members)))
            if len(group) >= 3:
                track_indices[root(next(iter(group.values())))] = len(self.tracks)
                self.tracks.append({f: self.pixels[node] for f, node in sorted(group.items())})
                if len(self.tracks) == 1 or len(self.tracks) % 20000 == 0:
                    show_track(group, f"{len(self.tracks):,} tracks built · {group_index+1:,}/{len(members):,} components processed")
            progress.advance()
        self.loop_links = {}
        for a, b in loop_edges:
            identifier = track_indices.get(root(a))
            if identifier is not None:
                self.loop_links.setdefault(identifier, set()).add((a.frame, b.frame))
        delta = np.zeros(self.count + 1, dtype=np.int64)
        temporal_tracks = 0
        for group in temporal_members.values():
            if len(group) >= 3:
                temporal_tracks += 1
                delta[min(group) + 1] += 1
                delta[max(group) + 1] -= 1
        self.temporal_crossing_counts = np.cumsum(delta)[:self.count]
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
                        "temporal_tracks": temporal_tracks,
                        "loop_closure": {"candidate_pairs": loop_candidates,
                            "geometrically_verified_pairs": loop_geometry,
                            "corroborated_pairs": corroborated_loops,
                            "consistent_loop_tracks": len(self.loop_links)},
                        "components": components, "frontend": "RaCo-ALIKED/LightGlue+; native F/H RANSAC"}
