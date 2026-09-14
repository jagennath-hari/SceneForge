"""Application orchestration for one inspectable geometry run."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

from stereoforge.utils.artifacts import staged_output, write_json
from stereoforge.utils.validation import require_finite, require_integer
from stereoforge.utils.progress import Progress
from stereoforge.utils.visualization import GeometryReportWriter
from stereoforge.video.sampling import VideoFrameSampler
from stereoforge.refinement.pycusfm import CuSFMRefiner
from stereoforge.refinement.report import write_refinement_report, write_refinement_status

from .config import DemoConfig
from .inputs import select_image_paths, validate_image_paths
from .types import DepthUnits
from .adaptive import AdaptiveGeometryEstimator

LOGGER = logging.getLogger(__name__)
CHECKPOINT_NAME = "vggt_omega_1b_512.pt"
CHECKPOINT_REPOSITORY = "facebook/VGGT-Omega"


@dataclass(frozen=True, slots=True)
class DemoRequest:
    output: Path
    checkpoint: Path | None = None
    images: Path | None = None
    video: Path | None = None
    frame_count: int | None = None
    start_seconds: float | None = None
    duration: float | None = None

    def __post_init__(self) -> None:
        for name in ("output", "checkpoint", "images", "video"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        if (self.images is None) == (self.video is None):
            raise ValueError("Supply exactly one image folder or video")
        if self.frame_count is not None:
            require_integer("frame_count", self.frame_count)
        if self.duration is not None:
            require_finite("duration", self.duration)
            if self.duration == 0:
                raise ValueError("duration must be positive")
        if self.video is not None:
            if self.start_seconds is not None:
                require_finite("start_seconds", self.start_seconds)
        elif self.start_seconds is not None or self.duration is not None:
            raise ValueError("--start-seconds and --duration only apply to --video")


@dataclass(frozen=True, slots=True)
class FrameSummary:
    frame_index: int
    valid_fraction: float
    median_depth: float | None


@dataclass(frozen=True, slots=True)
class DemoResult:
    output: Path
    units: DepthUnits
    frames: tuple[FrameSummary, ...]


class GeometryDemoRunner:
    """Coordinate inputs, checkpoint resolution, inference and report publication."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.report_writer = GeometryReportWriter(config.preview)

    def run(self, request: DemoRequest) -> DemoResult:
        self._validate_request(request)
        image_paths: tuple[Path, ...] = ()
        if request.images is not None:
            image_paths = select_image_paths(request.images, request.frame_count)
            with Progress("Validating input images"):
                validate_image_paths(image_paths)
        if request.video is not None and not request.video.is_file():
            raise FileNotFoundError(f"Video not found: {request.video}")
        with Progress("Resolving checkpoint"):
            checkpoint = self._resolve_checkpoint(request)
        with staged_output(request.output) as staging:
            timestamps: tuple[float, ...] | None = None
            video_extraction: dict[str, Any] | None = None
            if request.video is not None:
                LOGGER.info("Decoding continuous video; all frames are retained unless selection options are supplied")
                sampler = VideoFrameSampler(request.start_seconds or 0.0, request.duration, request.frame_count,
                                            max_edge=3 * self.config.geometry.image_resolution)
                sampled = sampler.sample(request.video, staging / "input_frames")
                image_paths, timestamps = sampled.paths, sampled.timestamps_seconds
                video_extraction = {
                    "source_size_wh": sampled.source_size, "working_size_wh": sampled.working_size,
                    "decoder": sampled.decoder, "max_edge": sampler.max_edge,
                    "note": "Working images are resized; intrinsics describe VGGT processed images, not source video pixels",
                }
                with Progress("Validating decoded images"):
                    validate_image_paths(image_paths)
            geometry = self.config.geometry
            if geometry.max_frames is not None and len(image_paths) > geometry.max_frames:
                raise ValueError(
                    f"Decoded {len(image_paths)} frames, exceeding configured max_frames={geometry.max_frames}. "
                    "Set max_frames: null to allow the complete clip. No frames were dropped."
                )
            LOGGER.info("Inferring geometry for all %d selected frames using adaptive overlapping sections and %s",
                        len(image_paths), checkpoint.name)
            sequence, reconstruction = AdaptiveGeometryEstimator(checkpoint, geometry).predict(
                image_paths, staging / "sections",
            )
            if request.video is not None:
                # Stored source names must refer to published files, not temporary paths.
                sequence = replace(sequence, source_names=tuple(
                    str(request.output / "input_frames" / path.name) for path in image_paths
                ))
            provenance: dict[str, Any] = {
                "checkpoint": str(checkpoint), "image_resolution": geometry.image_resolution,
                "preprocess_mode": geometry.preprocess_mode, "device": geometry.device,
                "video": str(request.video) if request.video else None,
                "video_timestamps_seconds": timestamps,
                "video_extraction": video_extraction,
                "frame_count": len(image_paths),
                "selection": {"start_seconds": request.start_seconds, "duration": request.duration,
                              "requested_frames": request.frame_count},
                "inference_mode": "adaptive_overlapping_sections",
                "reconstruction": reconstruction,
                "frame_index_note": "Indices are local to the selected sequence, not original video frames",
            }
            if self.config.refinement.enabled:
                provenance["refinement"] = {"status": "pending"}
            write_json(staging / "run.json", provenance)
            # Preserve the unrefined VGGT result independently of SfM success.
            metadata = self.report_writer.write(sequence, staging, provenance=provenance,
                                                viewer=not self.config.refinement.enabled)
            if self.config.refinement.enabled:
                try:
                    refinement = CuSFMRefiner(self.config.refinement).run(sequence, timestamps, staging / "pycusfm")
                    comparison = write_refinement_report(staging / "pycusfm", metadata, self.config.preview.max_points)
                    provenance["refinement"] = {**refinement.summary, "status": comparison.get("status", "complete"),
                                                "comparison": comparison}
                except (Exception, KeyboardInterrupt) as exc:
                    status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    message = str(exc) or "Refinement interrupted by user"
                    provenance["refinement"] = write_refinement_status(staging / "pycusfm", message, status)
                    LOGGER.warning("pyCuSFM %s: %s. Preserving source geometry and all refinement files.", status, message)
                if provenance["refinement"]["status"] not in {"complete", "failed", "interrupted"}:
                    LOGGER.warning("pyCuSFM returned a partial or empty reconstruction; inspect its report and logs")
                write_json(staging / "run.json", provenance)
                metadata["provenance"] = provenance
                write_json(staging / "metadata.json", metadata)
                # One visible reconstruction: the entry page opens the final
                # sparse viewer, or its diagnostic page if refinement failed.
                (staging / "index.html").write_text(
                    '<!doctype html><meta charset="utf-8">'
                    '<meta http-equiv="refresh" content="0; url=pycusfm/index.html">'
                    '<title>StereoForge reconstruction</title>'
                    '<a href="pycusfm/index.html">Open final reconstruction</a>', encoding="utf-8")
            summaries = tuple(FrameSummary(f["frame_index"], f["valid_fraction"], f["depth_p50"])
                              for f in metadata["frames"])
        # Only return success after the complete report is published.
        return DemoResult(request.output, sequence.units, summaries)

    def _validate_request(self, request: DemoRequest) -> None:
        limit = self.config.geometry.max_frames
        if limit is not None and request.frame_count is not None and request.frame_count > limit:
            raise ValueError(f"--frames exceeds configured max_frames={limit}")

    def _resolve_checkpoint(self, request: DemoRequest) -> Path:
        if request.checkpoint is not None:
            if not request.checkpoint.is_file():
                raise FileNotFoundError(f"Explicit checkpoint not found: {request.checkpoint}")
            return request.checkpoint
        if self.config.geometry.image_resolution != 512:
            raise ValueError("The automatic checkpoint requires resolution 512; supply --checkpoint for another model")
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        # Check disk without a HEAD request or credential/network requirement.
        try:
            return Path(hf_hub_download(repo_id=CHECKPOINT_REPOSITORY, filename=CHECKPOINT_NAME,
                                       local_files_only=True))
        except LocalEntryNotFoundError:
            pass
        LOGGER.info("Checkpoint not found locally; downloading to the persistent Hugging Face cache")
        try:
            return Path(hf_hub_download(repo_id=CHECKPOINT_REPOSITORY, filename=CHECKPOINT_NAME))
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                raise RuntimeError(
                    "Hugging Face denied checkpoint access. Confirm VGGT-Omega access was approved "
                    "for the token's account and that the container has its read token mounted."
                ) from exc
            raise RuntimeError("Checkpoint download failed; check the Hugging Face service and network") from exc
