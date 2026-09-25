"""CPU tensor artifacts for VGGT windows."""
from pathlib import Path
import torch
from .types import FrameGeometry, GeometrySequence

def save_sequence(sequence: GeometrySequence, path: Path) -> None:
    # Tensor-only dictionaries can be loaded with weights_only=True.
    torch.save({"frames": [dict(frame_index=f.frame_index, depth=f.depth.clone(),
                               confidence=f.confidence.clone(), intrinsics=f.intrinsics.clone(),
                               camera_to_world=f.camera_to_world.clone()) for f in sequence.frames],
                "rgb": sequence.processed_rgb, "names": sequence.source_names,
                "sizes": sequence.original_sizes_hw}, path)


def load_sequence(path: Path) -> GeometrySequence:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return GeometrySequence(tuple(FrameGeometry(**f) for f in data["frames"]),
                            data["rgb"], data["names"], data["sizes"])

