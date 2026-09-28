"""Replay one saved global BA stage with native cuNLS iteration logging."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stereoforge.reconstruction.native_map import native_backend
from stereoforge.utils.artifacts import write_json


class BAReplay:
    """Keep checkpoint states, observations, prior targets and solver policy intact."""

    def __init__(self, checkpoint: Path, stage: int, device: int) -> None:
        self.checkpoint = checkpoint.resolve()
        self.stage = stage
        self.device = device

    def run(self) -> Path:
        metadata = json.loads((self.checkpoint / f'stage_{self.stage}.json').read_text())
        if metadata['format_version'] != 1 or metadata['options']['use_gnc']:
            raise ValueError('Replay requires a version-1 non-GNC global BA checkpoint')
        with np.load(self.checkpoint / 'input.npz', allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        # Stage n starts from stage n-1, but every stage keeps the original priors.
        if self.stage > 1:
            with np.load(self.checkpoint / f'stage_{self.stage - 1}.npz', allow_pickle=False) as saved:
                for name in ('world_to_camera', 'intrinsics', 'points'):
                    arrays[name] = saved[name]
        backend = native_backend()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        output = self.checkpoint / f'replay_stage_{self.stage}_{stamp}'
        output.mkdir(exist_ok=False)
        write_json(output / 'request.json', {
            'checkpoint': str(self.checkpoint), 'stage': self.stage,
            'device': self.device, 'metadata': metadata,
            'policy': 'Unmodified saved objective, observations, priors and solver settings',
        })
        print(f'Replaying saved BA stage {self.stage}; solver log: {output / "solver.log"}', flush=True)
        try:
            document = backend.replay_ba(arrays, metadata, self.device, str(output / 'solver.log'))
            report = document.pop('metadata')
            report['state_max_absolute_change'] = {
                name: float(np.max(np.abs(document[name] - arrays[name])))
                for name in ('world_to_camera', 'intrinsics', 'points')
            }
            # Store raw squared errors in binary; JSON summaries exclude infinities.
            for label, mask in (
                ('all', np.ones(len(arrays['observation_camera']), dtype=bool)),
                ('loop', arrays.get('loop_observation_mask', np.zeros(len(arrays['observation_camera']), dtype=bool))),
            ):
                for when in ('before', 'after'):
                    errors = np.sqrt(document[f'squared_errors_{when}'][mask.astype(bool)])
                    finite = errors[np.isfinite(errors)]
                    report[f'{label}_{when}'] = {
                        'observations': int(errors.size),
                        'invalid': int(errors.size - finite.size),
                        'median_pixels': float(np.median(finite)) if finite.size else None,
                        'within_5px': int(np.count_nonzero(errors <= 5)),
                    }
            np.savez(output / 'result.npz', **document)
            write_json(output / 'report.json', report)
        except Exception as error:
            write_json(output / 'failure.json', {'error': str(error)})
            raise RuntimeError(f'BA replay failed; diagnostics retained in {output}') from error
        print(f'Replay finished; convergence is not implied. Inspect {output / "solver.log"}', flush=True)
        print(f'State changes: {report["state_max_absolute_change"]}', flush=True)
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Saved map_*/global_ba directory')
    parser.add_argument('--stage', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    try:
        BAReplay(args.checkpoint, args.stage, args.device).run()
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        parser.exit(1, f'ERROR: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
