"""Shared video reconstruction frontend."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import torch

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from stereoforge.video.sampling import VideoFrameSampler
from .images import FeatureImages, PreparedImage
from .retrieval import GlobalDescriptors, loop_pair_quality

if TYPE_CHECKING:
    from .windowed import WindowOptions


class ReconstructionFrontend:
    def __init__(self, options: WindowOptions, output: Path, keyframe_config: Path) -> None:
        self.options, self.output, self.keyframe_config = options, output, keyframe_config
        self.visualization = None
        self.retrieval: GlobalDescriptors | None = None
        self.loop_candidates: dict[tuple[int, int], float] = {}
        self.verified_loops: set[tuple[int, int]] = set()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for reconstruction matching and optimization")
        self.devices = ([f"cuda:{i}" for i in range(torch.cuda.device_count())] if options.device == "cuda"
                        else [options.device])
        self.device = int(self.devices[0].split(":")[1])
        if self.device >= torch.cuda.device_count():
            raise ValueError("Requested device is outside the visible CUDA devices")
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the reconstruction pair matcher")
        capabilities = subprocess.run([executable, "--capabilities"], capture_output=True, text=True)
        if capabilities.returncode or not json.loads(capabilities.stdout).get("stable_feature_ids"):
            raise RuntimeError("Pair matcher needs rebuilding: missing stable feature ID support")

    def _prepare_images(self, paths: list[Path], reused: dict[Path, Path] | None = None,
                        description: str = "Preparing feature images",
                        timestamps: tuple[float, ...] = ()) -> dict[int, Path]:
        folder = self.output / "processed"
        folder.mkdir(exist_ok=True)
        result, sizes = {}, set()
        self.loop_candidates = {}
        if self.visualization is not None:
            self.visualization.activity('Preparing feature images' + (' and global descriptors' if self.options.loop_closure else ''), overview=True)
        with Progress(description, len(paths), "frame") as progress:
            if self.options.loop_closure and self.retrieval is None:
                progress.status('preparing SelaVPR++ descriptor cache')
                self.retrieval = GlobalDescriptors(self.device)
            retrieval = self.retrieval if self.options.loop_closure else None
            descriptors = np.empty((len(paths), GlobalDescriptors.dimension), dtype=np.float32) if retrieval else None
            pending: list[PreparedImage] = []
            cached = 0

            def flush() -> None:
                if not pending or retrieval is None:
                    return
                progress.status(f'SelaVPR++: {len(pending)} new descriptors; {cached} reused')
                if descriptors is None:
                    raise RuntimeError('Descriptor storage is not initialized')
                batch: list[tuple[str, torch.Tensor]] = []
                for item in pending:
                    if item.key is None or item.retrieval_input is None:
                        raise RuntimeError('Missing prepared retrieval input')
                    batch.append((item.key, item.retrieval_input))
                values = retrieval.infer(batch)
                for item, value in zip(pending, values, strict=True):
                    descriptors[item.index] = value
                progress.advance(len(pending))
                pending.clear()

            try:
                for item in FeatureImages(folder, reused or {}, retrieval).iterate(paths):
                    result[item.index] = item.path
                    sizes.add(item.size)
                    if retrieval is None:
                        progress.advance()
                    elif item.descriptor is not None:
                        descriptors[item.index] = item.descriptor
                        cached += 1
                        progress.advance()
                    else:
                        pending.append(item)
                        if len(pending) == retrieval.options.batch_size:
                            flush()
                    if self.visualization is not None:
                        self.visualization.keyframe(item.index, item.path)
                        self.visualization.activity(f"Feature image ready: {item.index+1}/{len(paths)}", [item.index], ready=True)
                flush()
                if retrieval is not None:
                    # Free the descriptor network before loading matching engines.
                    retrieval.close()
                    progress.status('retrieving distant image pairs')
                    self.loop_candidates = retrieval.retrieve(descriptors, timestamps, self.options.neighbors)
                    write_json(self.output / 'retrieval.json', {**retrieval.document(),
                               'frames': len(paths), 'cached_descriptors': cached,
                               'candidate_pairs': len(self.loop_candidates)})
            finally:
                if retrieval is not None:
                    retrieval.close()
        if self.visualization is not None:
            self.visualization.event(f"Feature images ready: {len(paths)}/{len(paths)}; {len(self.loop_candidates)} loop candidates")
        if len(sizes) != 1:
            raise ValueError("Video input must use a uniform processed image size")
        self.size_wh = next(iter(sizes))
        return result

    def _match(self, count: int, reused: dict[tuple[int, int], dict] | None = None,
               description: str = "Verifying image pairs") -> list[Path]:
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the pair matcher with global feature IDs")
        folder = self.output / "matching"
        folder.mkdir(exist_ok=True)
        models = None
        # Reuse temporal evidence across inserted frames; loop proposals are
        # re-ranked using the current timestamps and descriptors.
        reused = {pair: record for pair, record in (reused or {}).items()
                  if record.get('pair_kind', 'temporal') == 'temporal' or pair in self.loop_candidates}
        prior_targets: dict[int, list[int]] = {}
        for a, b in set(reused) | set(self.loop_candidates):
            prior_targets.setdefault(a, []).append(b)
        result = []
        width, height = self.size_wh
        batches = range(0, max(count - 1, 0), 64)
        plans = []
        supported_loops = set()
        for start in batches:
            pairs = sorted({(a, b) for a in range(start, min(start+64, count))
                            for b in range(a+1, min(a+self.options.neighbors+1, count))}
                           | {(a, b) for a in range(start, min(start+64, count))
                              for b in prior_targets.get(a, ())})
            destination = folder / f"{start:06d}.jsonl"
            complete = False
            if destination.is_file():
                # A cache file is reusable only for this exact pair plan. On a
                # changed retrieval ranking, preserve measurements for retained
                # pairs and compute only newly requested ones.
                with destination.open() as stream:
                    cached_records = [json.loads(line) for line in stream]
                complete = ([(r['source'], r['target']) for r in cached_records] == pairs and
                            all('pair_kind' in r and (r['pair_kind'] != 'loop' or 'loop_geometry' in r)
                                for r in cached_records))
                if complete:
                    supported_loops.update((r['source'], r['target']) for r in cached_records
                        if r['pair_kind'] == 'loop' and r['loop_geometry']['accepted'])
                else:
                    requested = set(pairs)
                    reused.update({(r['source'], r['target']): r for r in cached_records
                                   if (r['source'], r['target']) in requested})
                del cached_records
            missing = [] if complete else [pair for pair in pairs if pair not in reused]
            plans.append((start, destination, pairs, missing, complete))
        # Engine preparation has its own bar. Finish it before opening the
        # matching bar, including when only a middle batch needs new inference.
        if any(missing for _, _, _, missing, _ in plans):
            if self.visualization is not None:
                self.visualization.event("Preparing RaCo–ALIKED + LightGlue+ for new image pairs")
            models = VideoFrameSampler.model_arguments(self.keyframe_config)[1]
        total_new = sum(len(missing) for _, _, _, missing, _ in plans)
        total_pairs = sum(len(pairs) for _, _, pairs, _, _ in plans)
        completed_new = 0
        with Progress(description, total=total_pairs, unit="pair") as progress:
            for batch_index, (start, destination, pairs, missing, complete) in enumerate(plans):
                progress.status(f"batch {batch_index+1}/{len(plans)} · new {completed_new}/{total_new} · "
                                f"{total_pairs-total_new} reusable")
                if complete:
                    result.append(destination)
                    if self.visualization is not None:
                        self.visualization.matched_batch(destination)
                    progress.advance(len(pairs))
                    continue
                if not pairs:
                    continue
                request = folder / f"{start:06d}.request.json"
                write_json(request, {"pairs": [{"source": a, "target": b, "width": width, "height": height,
                             "source_path": str(self.images[a]), "target_path": str(self.images[b])} for a, b in missing]})
                temporary = destination.with_suffix(".partial_" + uuid4().hex[:8])
                command = [executable, str(request), str(self.keyframe_config), models or "", str(temporary)]
                if self.device:
                    command.append(str(self.device))
                if self.visualization is not None:
                    sample = pairs[:16]
                    self.visualization.activity(
                        f"Matching batch {start//64+1}/{len(batches)} — requested pairs, awaiting verification",
                        sorted({f for pair in sample for f in pair}), sample)
                if missing:
                    with destination.with_suffix(".log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Verify identity and schema before marking this batch reusable.
                records = [json.loads(line) for line in temporary.read_text().splitlines()] if missing else []
                if [(r["source"], r["target"]) for r in records] != missing:
                    raise ValueError("Pair matcher returned incomplete or unexpected frame IDs")
                if any("source_feature" not in m for r in records for m in r["matches"]):
                    raise ValueError("Rebuild Docker: pair matcher lacks stable global feature IDs")
                combined = {pair: reused[pair] for pair in pairs if pair in reused}
                combined.update({(record["source"], record["target"]): record for record in records})
                for pair, record in combined.items():
                    temporal = (pair[1]-pair[0] <= self.options.neighbors or
                                (pair in reused and reused[pair].get('pair_kind', 'temporal') == 'temporal'))
                    record['pair_kind'] = 'temporal' if temporal else 'loop'
                    if not temporal:
                        record['retrieval_score'] = self.loop_candidates[pair]
                        record['loop_geometry'] = loop_pair_quality(record, width, height)
                        if record['loop_geometry']['accepted']:
                            supported_loops.add(pair)
                with temporary.open("w") as stream:
                    for pair in pairs:
                        stream.write(json.dumps(combined[pair]) + "\n")
                temporary.rename(destination)
                result.append(destination)
                if self.visualization is not None:
                    self.visualization.matched_batch(destination)
                completed_new += len(missing)
                progress.advance(len(pairs))
            progress.status('checking neighboring loop-pair support')
            self.verified_loops = {pair for pair in supported_loops if any(
                (pair[0]+da, pair[1]+db) in supported_loops
                for da in (-2, -1, 1, 2) for db in (-2, -1, 1, 2))}
            # Publish only compact decisions. Do not reread/rewrite every large
            # match file merely to annotate the cross-batch corroboration result.
            write_json(folder / 'loop_verification.json', {
                'candidate_pairs': len(self.loop_candidates),
                'geometry_verified_pairs': sorted(supported_loops),
                'accepted_pairs': sorted(self.verified_loops),
                'criteria': '30 inliers, 25% inlier ratio, 4/16 cells in each image; another verified pair within 2 frames on both sides'})
        return result
