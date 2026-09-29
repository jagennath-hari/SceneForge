"""Resolve an authorized checkpoint from disk or the gated Hugging Face cache."""
import logging
from pathlib import Path
LOGGER = logging.getLogger(__name__)
CHECKPOINT_NAME = "vggt_omega_1b_512.pt"
CHECKPOINT_REPOSITORY = "facebook/VGGT-Omega"


def resolve_checkpoint(checkpoint: Path | None = None) -> Path:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Explicit checkpoint not found: {checkpoint}")
        return checkpoint
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
                "Hugging Face denied checkpoint access. Confirm VGGT-Ω access was approved "
                "for the token's account and that the container has its read token mounted."
            ) from exc
        raise RuntimeError("Checkpoint download failed; check the Hugging Face service and network") from exc
