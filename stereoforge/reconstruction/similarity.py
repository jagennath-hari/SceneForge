"""Fit a scale-aware similarity from corresponding 3D landmarks."""
import numpy as np

def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity fitting requires at least three finite point pairs")
    a, b = source.mean(axis=0), target.mean(axis=0)
    x, y = source - a, target - b
    try:
        u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Overlap similarity fit did not converge") from exc
    if singular[0] <= 0 or singular[1] < singular[0] * 1e-6:
        raise ValueError("Overlap geometry is degenerate; cannot determine a reliable alignment")
    sign = np.ones(3)
    sign[-1] = -1 if np.linalg.det(u @ vt) < 0 else 1
    rotation = (u * sign) @ vt
    scale = float((singular * sign).sum() / np.mean(np.sum(x * x, axis=1)))
    translation = b - scale * (rotation @ a)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
        raise ValueError("Overlap produced an invalid similarity transform")
    return scale, rotation, translation

