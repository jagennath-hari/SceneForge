"""Rebuild a WebGL viewer from saved geometry without rerunning inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from stereoforge.geometry.config import PreviewConfig
from .visualization import GeometryArrays, GeometryReportWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing geometry output directory")
    parser.add_argument("--max-points", type=int, default=240000)
    args = parser.parse_args()
    writer = GeometryReportWriter(PreviewConfig(max_points=args.max_points))
    metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    with np.load(args.run / "geometry.npz", allow_pickle=False) as data:
        arrays = GeometryArrays(data["depth"], data["confidence"], data["valid_mask"],
                                data["intrinsics"], data["camera_to_world"], data["processed_rgb"],
                                tuple(f["confidence_threshold"] for f in metadata["frames"]))
    xyz, colors, point_frames = writer._sample_points(arrays)
    writer._write_viewer(args.run, metadata, xyz, colors, point_frames)
    print(f"Updated {args.run / 'index.html'}")


if __name__ == "__main__":
    main()
