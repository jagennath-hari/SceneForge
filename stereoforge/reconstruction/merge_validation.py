"""Withhold measured overlap observations from each provisional merge's BA."""

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from stereoforge.refinement.bundle_adjustment import require_connected
from stereoforge.refinement.sparse_model import SparseModel, reprojection_error
from .registration import shared_landmarks

POLICY = "camera_aware_sim3_retriangulation_holdout_v3"


def observation_key(frame: int, uv: np.ndarray) -> tuple[int, float, float]:
    return frame, float(round(uv[0], 4)), float(round(uv[1], 4))


def model_fingerprint(model: SparseModel) -> str:
    digest = hashlib.sha256()
    for frame, camera in sorted(model.cameras.items()):
        digest.update(json.dumps([frame, camera.size_hw]).encode())
        digest.update(np.asarray(camera.pose, dtype="<f8").tobytes())
        digest.update(np.asarray(camera.intrinsics, dtype="<f8").tobytes())
    for point in model.points:
        digest.update(np.asarray(point.xyz, dtype="<f8").tobytes())
        digest.update(np.asarray(point.rgb, dtype=np.uint8).tobytes())
        for frame, uv in sorted(point.observations.items()):
            digest.update(json.dumps([frame, *map(float, uv)]).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MergeValidation:
    training: SparseModel
    observations: tuple[dict, ...]
    shared_frames: tuple[int, ...]

    @classmethod
    def prepare(cls, seed: SparseModel, reference: SparseModel, local: SparseModel) -> "MergeValidation":
        shared = tuple(sorted(reference.cameras.keys() & local.cameras.keys()))
        joined_keys = {observation_key(f, uv)
                       for i, _ in shared_landmarks(reference, local)
                       for f, uv in reference.points[i].observations.items()}
        support = Counter(f for p in seed.points for f in p.observations)
        held_counts: Counter[int] = Counter()
        points, held = [], []
        for point in seed.points:
            candidates = [f for f in point.observations if f in shared and support[f] > 10]
            joined = any(observation_key(f, uv) in joined_keys for f, uv in point.observations.items())
            if not joined or len(point.observations) < 4 or not candidates:
                points.append(point)
                continue
            # Selection uses observation identity/support, never post-BA residuals.
            frame = min(candidates, key=lambda f: (held_counts[f], f))
            remaining = {f: uv for f, uv in point.observations.items() if f != frame}
            held.append({"frame": frame, "uv": point.observations[frame].tolist(),
                         "training_keys": [observation_key(f, uv) for f, uv in sorted(remaining.items())]})
            points.append(replace(point, observations=remaining))
            support[frame] -= 1
            held_counts[frame] += 1
        if len(held) < 40 or any(held_counts[f] < 5 for f in shared):
            raise ValueError("Insufficient overlap observations for joint-BA validation: "
                             f"{len(held)} total, per-camera counts {dict(sorted(held_counts.items()))}; "
                             "need 40 total and five per shared camera")
        training = SparseModel(seed.cameras, tuple(points))
        require_connected(training)
        return cls(training, tuple(held), shared)

    def document(self) -> dict:
        return {"policy": POLICY, "shared_frames": self.shared_frames,
                "scope": "Withheld from this merge BA only; used in child BA and initialization. Seed filtering precedes selection.",
                "observations": self.observations}

    def evaluate(self, optimized: SparseModel) -> dict:
        lookup: dict[tuple, set[int]] = {}
        for index, point in enumerate(optimized.points):
            for frame, uv in point.observations.items():
                lookup.setdefault(observation_key(frame, uv), set()).add(index)
        records = []
        for held in self.observations:
            votes: Counter[int] = Counter()
            for key in held["training_keys"]:
                votes.update(lookup.get(tuple(key), ()))
            # Native BA may renumber landmarks. Require at least two retained
            # measured observations to identify one unambiguous optimized point.
            index = next(iter(votes)) if len(votes) == 1 and next(iter(votes.values())) >= 2 else None
            frame = held["frame"]
            error = float("inf")
            if index is not None and frame in optimized.cameras:
                error = reprojection_error(optimized.cameras[frame], optimized.points[index].xyz,
                                           np.asarray(held["uv"]))
            records.append({"frame": frame, "error_pixels": float(error) if np.isfinite(error) else None,
                            "passed": bool(error <= 5)})
        frames = [{"frame": frame, "count": sum(r["frame"] == frame for r in records),
                   "within_5px": float(np.mean([r["passed"] for r in records if r["frame"] == frame]))}
                  for frame in self.shared_frames]
        fraction = float(np.mean([r["passed"] for r in records]))
        lost = sorted(self.training.cameras.keys() - optimized.cameras.keys())
        accepted = not lost and fraction >= 0.8 and all(f["within_5px"] >= 0.8 for f in frames)
        return {"policy": POLICY, "status": "accepted" if accepted else "rejected",
                "held_out_count": len(records), "within_5px": fraction, "frames": frames,
                "lost_supported_cameras": lost, "observations": records,
                "missing_landmarks_count_as_failures": True}
