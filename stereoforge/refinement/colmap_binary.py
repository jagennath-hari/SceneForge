"""Convert our PINHOLE COLMAP seeds to standard little-endian binary input.

The pinned cuSFM text point reader extracts RGB with operator>>(char&), which
breaks numeric color tokens and the following tracks. Keep text for inspection,
but pass a separate binary-only directory to the native optimizer.
"""

from collections.abc import Iterator
from pathlib import Path
import struct


class ColmapBinaryWriter:
    @staticmethod
    def _records(path: Path) -> Iterator[list[str]]:
        with path.open() as stream:
            for line in stream:
                if line.strip() and not line.startswith("#"):
                    yield line.split()

    def convert(self, source: Path, destination: Path) -> None:
        destination.mkdir()
        cameras = list(self._records(source / "cameras.txt"))
        with (destination / "cameras.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", len(cameras)))
            for row in cameras:
                if len(row) != 8 or row[1] != "PINHOLE":
                    raise ValueError("Binary seed conversion requires PINHOLE cameras")
                stream.write(struct.pack("<IiQQdddd", int(row[0]), 1, int(row[2]), int(row[3]),
                                         *map(float, row[4:])))
        # Patch counts at EOF to avoid retaining all observations in memory.
        with (source / "images.txt").open() as text, (destination / "images.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 0))
            count = 0
            while line := text.readline():
                if not line.strip() or line.startswith("#"):
                    continue
                row = line.split()
                if len(row) != 10 or "\x00" in row[9]:
                    raise ValueError("Malformed COLMAP seed image")
                stream.write(struct.pack("<I7dI", int(row[0]), *map(float, row[1:8]), int(row[8])))
                stream.write(row[9].encode("utf-8") + b"\x00")
                line = text.readline()
                if line == "":
                    raise ValueError("Missing COLMAP observation line")
                values = line.split()
                if len(values) % 3:
                    raise ValueError("Malformed COLMAP seed observations")
                stream.write(struct.pack("<Q", len(values)//3))
                for i in range(0, len(values), 3):
                    stream.write(struct.pack("<ddq", float(values[i]), float(values[i+1]), int(values[i+2])))
                count += 1
            stream.seek(0)
            stream.write(struct.pack("<Q", count))
        with (destination / "points3D.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 0))
            count = 0
            for row in self._records(source / "points3D.txt"):
                if len(row) < 8 or (len(row)-8) % 2:
                    raise ValueError("Malformed COLMAP seed landmark")
                stream.write(struct.pack("<QdddBBBdQ", int(row[0]), *map(float, row[1:4]),
                                         *map(int, row[4:7]), float(row[7]), (len(row)-8)//2))
                for i in range(8, len(row), 2):
                    stream.write(struct.pack("<II", int(row[i]), int(row[i+1])))
                count += 1
            stream.seek(0)
            stream.write(struct.pack("<Q", count))


def bundle_adjustment_command(package: Path, directory: Path) -> list[str]:
    return [str(package / "bin/bundle_adjustment_runner"),
            "--colmap_sparse_dir", str(directory / "initialized_binary"),
            "--output_dir", str(directory / "workspace/ba_raw"),
            "--output_binary=false", "--iterative_mode=false", "--redo_triangulation=false",
            "--max_iterations=200", "--fix_first_camera=true",
            "--loss_function=CAUCHY", "--use_cudss_solver=true", "--verbose=true"]
