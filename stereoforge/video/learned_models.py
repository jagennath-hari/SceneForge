"""Prepare the split LightGlue-ONNX models and local TensorRT engine cache.

Python is used only for export/build. Frame extraction and matching run in C++.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn

from stereoforge.utils.progress import Progress

UPSTREAM_REVISION = "d12b4ba1632f558234e3f084e1f3d8bdf9147890"


class SuperPointExport(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        from lightglue_dynamo.models.superpoint import SuperPoint
        self.extractor = SuperPoint(num_keypoints=features).eval()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        points, scores, descriptors = self.extractor(image)
        return points.float(), scores, descriptors


class LightGlueExport(nn.Module):
    """Upstream fixed-depth network with fixed-size mutual-match outputs.

    Separate frame inputs bind to retained device allocations. Avoid
    data-dependent NonZero output allocations in the native runtime. The
    learned assignment and mutual nearest selection are unchanged; C++ applies
    the configurable confidence threshold and detector validity mask afterward.
    """

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self.register_buffer("canvas", torch.tensor([width, height], dtype=torch.float32))
        from lightglue_dynamo.models.lightglue import LightGlue
        self.matcher = LightGlue(
            url="https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
            depth_confidence=-1, width_confidence=-1,
        ).eval()

    def forward(self, points0: torch.Tensor, descriptors0: torch.Tensor,
                points1: torch.Tensor, descriptors1: torch.Tensor) -> tuple[torch.Tensor, ...]:
        keypoints = 2 * torch.cat((points0, points1), dim=0) / self.canvas - 1
        descriptors = torch.cat((descriptors0, descriptors1), dim=0)
        encoded = self.matcher.posenc(keypoints)
        features = self.matcher.input_proj(descriptors)
        for layer in self.matcher.transformers:
            features = layer(features, encoded)
        assignments = self.matcher.log_assignment[-1](features)
        values, targets = assignments.max(dim=2)
        sources = assignments.argmax(dim=1)
        reference = torch.arange(targets.shape[1], device=targets.device).expand_as(targets)
        mutual = sources.gather(1, targets) == reference
        indices = torch.where(mutual, targets, -torch.ones_like(targets))
        return indices.int(), values.exp()


@dataclass(frozen=True, slots=True)
class LearnedModelCache:
    features: int
    width: int
    height: int
    precision: str = "fp32"

    @classmethod
    def from_settings(cls, settings: dict) -> LearnedModelCache:
        values = (settings.get("features", 1024), settings.get("model_width", 960),
                  settings.get("model_height", 544))
        if any(type(value) is not int for value in values):
            raise ValueError("Learned model dimensions and feature count must be integers")
        features, width, height = values
        if not 120 <= features <= 4096 or any(value < 64 or value > 4096 or value % 8 for value in (width, height)):
            raise ValueError("Use 120–4096 keypoints and model dimensions divisible by eight, between 64 and 4096")
        precision = settings.get("precision", "fp32")
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be fp32 or fp16; INT8 requires a separate calibration workflow")
        return cls(features, width, height, precision)

    def prepare(self) -> Path:
        import tensorrt as trt
        if not torch.cuda.is_available():
            raise RuntimeError("SuperPoint/LightGlue requires a visible CUDA GPU")
        # A serialized engine is local to the runtime/hardware it was built for.
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"name": properties.name, "capability": [properties.major, properties.minor],
                            "memory": properties.total_memory, "uuid": str(properties.uuid)})
        driver_file = Path("/proc/driver/nvidia/version")
        driver = driver_file.read_text() if driver_file.is_file() else "unavailable"
        identity = {"upstream": UPSTREAM_REVISION, "driver": driver, "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "torch": torch.__version__, "tensorrt": trt.__version__, "devices": devices,
                    "features": self.features, "width": self.width, "height": self.height,
                    "precision": self.precision, "onnx": self._onnx_identity(),
                    "build": {"optimization_level": 5, "workspace_mib": 12288,
                              "tactic_shared_memory": "device_maximum", "sparsity": "existing_weights_only",
                              "float_io": self.precision, "indices_io": "int32", "runtime": "full"}}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "stereoforge/keyframes"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / key
        with (root / f"{key}.lock").open("a") as lock, Progress("Preparing SuperPoint + LightGlue") as progress:
            progress.status("checking model cache")
            fcntl.flock(lock, fcntl.LOCK_EX)
            onnx_directory = self.prepare_onnx(progress)
            if self._valid(destination, identity):
                return destination
            temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
            try:
                for index in range(len(devices)):
                    with torch.cuda.device(index):
                        for name in ("superpoint", "lightglue"):
                            progress.status(f"building {name} {self.precision} engine for cuda:{index}; first use only")
                            self._build(onnx_directory / f"{name}.onnx", temporary / f"{name}_{index}.engine")
                hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in temporary.glob("*.engine")}
                (temporary / "manifest.json").write_text(json.dumps({"identity": identity, "sha256": hashes}, indent=2))
                if destination.exists():
                    shutil.rmtree(destination)
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
                for index in range(len(devices)):
                    with torch.cuda.device(index):
                        torch.cuda.empty_cache()
        return destination

    def _onnx_identity(self) -> dict:
        # Precision, TensorRT, driver and GPU identities deliberately do not affect ONNX.
        return {"upstream": UPSTREAM_REVISION, "contract": "split_device_v2",
                "exporter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__, "features": self.features,
                "width": self.width, "height": self.height}

    def prepare_onnx(self, progress: Progress) -> Path:
        identity = self._onnx_identity()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "stereoforge/keyframes/onnx"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / key
        with (root / f"{key}.lock").open("a") as lock:
            progress.status("checking ONNX cache")
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                manifest = json.loads((destination / "manifest.json").read_text())
                valid = manifest["identity"] == identity and all(
                    hashlib.sha256((destination / name).read_bytes()).hexdigest() == manifest["sha256"][name]
                    for name in ("superpoint.onnx", "lightglue.onnx"))
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if valid:
                progress.status("reusing ONNX models")
                return destination
            temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
            try:
                progress.status("downloading missing weights and exporting ONNX models")
                self._export(temporary)
                hashes = {name: hashlib.sha256((temporary / name).read_bytes()).hexdigest()
                          for name in ("superpoint.onnx", "lightglue.onnx")}
                (temporary / "manifest.json").write_text(json.dumps({"identity": identity, "sha256": hashes}, indent=2))
                if destination.exists():
                    shutil.rmtree(destination)
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return destination

    @staticmethod
    def _valid(directory: Path, identity: dict) -> bool:
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            names = {f"{name}_{index}.engine" for index in range(len(identity["devices"]))
                     for name in ("superpoint", "lightglue")}
            return (manifest["identity"] == identity and set(manifest["sha256"]) == names and
                    all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == manifest["sha256"][name]
                        for name in names))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _export(self, directory: Path) -> None:
        # Export static inputs with standard ONNX operators, without ORT-specific
        # attention fusions. TensorRT parses these directly. No FP16/FP8 approximation.
        with torch.inference_mode():
            torch.onnx.export(SuperPointExport(self.features).eval(),
                              (torch.zeros(1, 1, self.height, self.width),), directory / "superpoint.onnx",
                              input_names=["image"], output_names=["keypoints", "scores", "descriptors"],
                              opset_version=17, dynamo=False)
            torch.onnx.export(LightGlueExport(self.width, self.height).eval(),
                              (torch.zeros(1, self.features, 2), torch.zeros(1, self.features, 256),
                               torch.zeros(1, self.features, 2), torch.zeros(1, self.features, 256)),
                              directory / "lightglue.onnx", input_names=["points0", "descriptors0", "points1", "descriptors1"],
                              output_names=["indices", "confidence"], opset_version=17, dynamo=False)

    def _build(self, source: Path, destination: Path) -> None:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(logger) as builder, builder.create_network(0) as network, \
                trt.OnnxParser(network, logger) as parser, builder.create_builder_config() as config:
            if not parser.parse_from_file(str(source)):
                errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
                raise RuntimeError(f"TensorRT could not parse {source.name}:\n{errors}")
            config.clear_flag(trt.BuilderFlag.TF32)
            if self.precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                # Keep sensitive normalization/score arithmetic in FP32. External
                # tensors use the selected precision; integer indices retain INT32.
                sensitive = {trt.LayerType.REDUCE, trt.LayerType.SOFTMAX, trt.LayerType.NORMALIZATION,
                             trt.LayerType.ELEMENTWISE, trt.LayerType.UNARY}
                for index in range(network.num_layers):
                    layer = network.get_layer(index)
                    if layer.type in sensitive and any(
                            layer.get_output(out).dtype == trt.float32 for out in range(layer.num_outputs)):
                        layer.precision = trt.float32
                        for out in range(layer.num_outputs):
                            if layer.get_output(out).dtype == trt.float32:
                                layer.set_output_type(out, trt.float32)
            # Equivalent build settings to the aggressive trtexec profile. Runtime
            # graph capture/spin waiting are handled by the C++ execution path.
            config.builder_optimization_level = 5
            sources = config.get_tactic_sources()
            for name in ("CUBLAS", "CUBLAS_LT", "CUDNN", "EDGE_MASK_CONVOLUTIONS", "JIT_CONVOLUTIONS"):
                source_type = getattr(trt.TacticSource, name, None)
                if source_type is not None:
                    sources |= 1 << int(source_type)
            if not config.set_tactic_sources(sources):
                raise RuntimeError("TensorRT rejected the requested tactic sources")
            # TACTIC_SHARED_MEMORY defaults to the device maximum per block.
            # An 800 MiB cap cannot increase the hardware's shared memory.
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12288 << 20)
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # Only exploit existing 2:4 sparsity; never prune weights.
            for index in range(network.num_inputs):
                tensor = network.get_input(index)
                if tensor.dtype == trt.float32 and self.precision == "fp16":
                    tensor.dtype = trt.float16
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            for index in range(network.num_outputs):
                tensor = network.get_output(index)
                if tensor.dtype == trt.float32 and self.precision == "fp16":
                    tensor.dtype = trt.float16
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError(f"TensorRT engine build failed for {source.name}; see diagnostics above")
            payload = bytes(serialized)
            # Inspect the built artifact, not just the requested builder flags.
            # This validates I/O types without running model inference.
            with trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(payload)
                if engine is None:
                    raise RuntimeError("Cannot inspect the built TensorRT engine")
                io_types = {}
                for index in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(index)
                    actual = engine.get_tensor_dtype(name)
                    expected = trt.int32 if name == "indices" else (trt.float16 if self.precision == "fp16" else trt.float32)
                    if actual != expected:
                        raise RuntimeError(f"Engine I/O mismatch for {name}: {actual}, expected {expected}")
                    io_types[name] = {"dtype": str(actual), "shape": list(engine.get_tensor_shape(name))}
                del engine
            destination.write_bytes(payload)
            destination.with_suffix(".io.json").write_text(json.dumps(io_types, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare cached SuperPoint/LightGlue ONNX models and TensorRT engines")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/keyframes_superpoint.json")
    parser.add_argument("--precision", choices=("fp32", "fp16"),
                        help="Override config for this build; use the same precision in the demo config")
    parser.add_argument("--onnx-only", action="store_true", help="Download weights/export ONNX without building GPU engines")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if settings.get("frontend") != "superpoint_lightglue":
            raise ValueError("Select a SuperPoint/LightGlue configuration")
        if args.precision:
            settings["precision"] = args.precision
        cache = LearnedModelCache.from_settings(settings)
        if args.onnx_only:
            with Progress("Preparing ONNX models") as progress:
                directory = cache.prepare_onnx(progress)
        else:
            directory = cache.prepare()
        print(f"Models ready: {directory}")
        if not args.onnx_only:
            print(f"TensorRT precision: {cache.precision}. Demos reuse these engines with matching config settings.")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Preparation stopped; completed cache entries are retained.\n")


if __name__ == "__main__":
    raise SystemExit(main())
