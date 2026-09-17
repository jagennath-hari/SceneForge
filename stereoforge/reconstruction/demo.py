"""End-to-end experimental hierarchical reconstruction of a continuous video."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
from uuid import uuid4

from stereoforge.geometry.config import DemoConfig
from stereoforge.geometry.inputs import validate_image_paths
from stereoforge.geometry.runner import DemoRequest, GeometryDemoRunner
from stereoforge.refinement.report import write_refinement_status
from stereoforge.utils.artifacts import write_json
from stereoforge.video.sampling import VideoFrameSampler
from .pipeline import HierarchicalReconstructor, HierarchyOptions

ROOT = Path(__file__).resolve().parents[2]


def signature(video: Path, keyframe_config: Path) -> dict:
    digest = hashlib.sha256()
    files = [*Path(__file__).parent.glob("*.py"), *(ROOT / "stereoforge/refinement").glob("*.py"),
             ROOT / "stereoforge/geometry/vggt_omega.py", ROOT / "stereoforge/geometry/alignment.py"]
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    executable = shutil.which("stereoforge-match-pairs")
    if executable:
        digest.update(Path(executable).read_bytes())
    stat = video.stat()
    return {"video": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "implementation_sha256": digest.hexdigest(), "keyframe_config": keyframe_config.read_text()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--resume", type=Path, help="Reuse completed stages of an unchanged hierarchical run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=None, help="cuda uses all GPUs for VGGT; cuda:N uses one")
    parser.add_argument("--cluster-size", type=int, default=None)
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--keyframe-config", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    output = None
    try:
        if args.resume:
            if any(value is not None for value in (args.output, args.checkpoint, args.device,
                    args.cluster_size, args.neighbors, args.keyframe_config)):
                raise ValueError("--resume uses saved settings; do not supply new configuration flags")
            candidate = args.resume.expanduser().resolve()
            saved = json.loads((candidate / "request.json").read_text())
            video, config = Path(saved["video"]), Path(saved["keyframe_config"])
            current = signature(video, config)
            immutable = ("video", "bytes", "mtime_ns", "keyframe_config")
            if any(current[key] != saved["signature"][key] for key in immutable):
                raise ValueError("Video or keyframe configuration changed; use a new output run")
            if current["implementation_sha256"] != saved["signature"]["implementation_sha256"]:
                logging.info("Code changed: reusing saved inputs and leaf BA; merge caches require the current validation policy")
            options = HierarchyOptions(**saved["options"])
            checkpoint = Path(saved["checkpoint"])
            if checkpoint.stat().st_size != saved["checkpoint_bytes"] or checkpoint.stat().st_mtime_ns != saved["checkpoint_mtime_ns"]:
                raise ValueError("Checkpoint changed; use a new output run")
            output = candidate
        else:
            video = args.video.expanduser().resolve()
            if not video.is_file():
                raise FileNotFoundError(f"Video not found: {video}")
            config = (args.keyframe_config or ROOT / "configs/keyframes_raco.json").expanduser().resolve()
            options = HierarchyOptions(cluster_size=args.cluster_size if args.cluster_size is not None else 32,
                                       neighbors=args.neighbors if args.neighbors is not None else 4,
                                       device=args.device or "cuda")
            candidate = (args.output or ROOT / "data/intermediate" / datetime.now(timezone.utc).strftime(
                "hierarchical_%Y%m%d_%H%M%S_%f")).expanduser().resolve()
            candidate.mkdir(parents=True, exist_ok=False)
            output = candidate
            checkpoint = GeometryDemoRunner(DemoConfig())._resolve_checkpoint(
                DemoRequest(output=output, video=video, checkpoint=args.checkpoint))
            write_json(output / "request.json", {"video": str(video), "keyframe_config": str(config),
                       "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                       "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                       "signature": signature(video, config), "options": asdict(options)})
        reconstructor = HierarchicalReconstructor(options, output, config)
        sampler = VideoFrameSampler(keyframe_config=config)
        folder = output / "input_frames"
        if (folder / "manifest.json").is_file():
            sampled = sampler._read_manifest(folder)
        else:
            if folder.exists():
                folder.rename(folder.with_name("input_frames.failed_" + uuid4().hex[:8]))
            sampled = sampler.sample(video, folder)
        validate_image_paths(sampled.paths)
        write_json(output / "selection.json", {"frames": len(sampled.paths),
                   "decoded_candidates": sampled.candidate_frame_count,
                   "source_frame_indices": sampled.source_frame_indices,
                   "timestamps_seconds": sampled.timestamps_seconds})
        result = reconstructor.run(list(sampled.paths), checkpoint, sampled.timestamps_seconds)
        write_json(output / "status.json", {"status": result["status"]})
        logging.info("%s: %d/%d frames. Open %s", result["status"], result["registered_frames"],
                     result["input_frames"], output / "index.html")
        return 0 if result["status"] == "complete" else 1
    except (Exception, KeyboardInterrupt) as exc:
        if output is not None:
            write_refinement_status(output, str(exc) or "Interrupted")
            links = [*output.glob("*.json"), *output.glob("nodes/*/*.json"),
                     *output.glob("nodes/*/ba/*.log"), *output.glob("matching/*.log"),
                     *output.glob("nodes/*/ba/index.html"), *output.glob("final/*.log")]
            with (output / "index.html").open("a") as stream:
                stream.write("<h2>Hierarchical stage diagnostics</h2><ul>")
                for path in sorted(links):
                    relative = path.relative_to(output).as_posix()
                    stream.write(f'<li><a href="{relative}">{relative}</a></li>')
                stream.write("</ul>")
            logging.error("%s\nArtifacts retained in %s", str(exc) or "Interrupted", output)
        else:
            logging.error("%s", exc)
        if args.debug:
            logging.exception("Hierarchical reconstruction traceback")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
