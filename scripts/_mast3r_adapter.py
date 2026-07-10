"""
Adapter for MASt3R pairwise 3D matching.

Wraps the official MASt3R inference pipeline and exposes a clean
``match_pair(…)`` interface that returns filtered 3D-3D correspondences
ready for Sim(3) estimation.

The locked MASt3R version is recorded in ``MAST3R_VERSION`` below.
If the official API surface changes, update this module accordingly.

Reference: https://github.com/naver/mast3r
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Locked MASt3R commit and checkpoint.
MAST3R_COMMIT = "HEAD"  # replaced with actual SHA when the repo is cloned
MAST3R_CHECKPOINT_DEFAULT = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


def _setup_mast3r_path() -> None:
    """Add third_party/mast3r to sys.path.

    The mast3r package internally imports ``mast3r.utils.path_to_dust3r``
    which adds its own dust3r submodule to sys.path. We must NOT also add
    the standalone ``third_party/dust3r`` to avoid shadowing the submodule.
    """
    project_root = Path(__file__).resolve().parent.parent
    mast3r_root = project_root / "third_party" / "mast3r"

    if str(mast3r_root) not in sys.path:
        sys.path.insert(0, str(mast3r_root))


def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint name or path to an actual file."""
    if Path(checkpoint).is_file():
        return checkpoint

    # Search common locations
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "third_party" / "LongSplat" / "submodules" / "mast3r" / "checkpoints" / checkpoint,
        project_root / "third_party" / "mast3r" / "checkpoints" / checkpoint,
        project_root / "checkpoints" / checkpoint,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Fallback: assume the path is directly loadable (e.g. Hugging Face hub)
    return checkpoint


class InsufficientMatchesError(Exception):
    """Raised when a segment pair has too few valid 3D correspondences."""

    def __init__(self, pair_key: str, num_matches: int, min_required: int = 3) -> None:
        super().__init__(
            f"Pair {pair_key}: only {num_matches} match(es) "
            f"(need >= {min_required} non-collinear correspondences)"
        )
        self.pair_key = pair_key
        self.num_matches = num_matches
        self.min_required = min_required


class Mast3rMatcher:
    """Pairwise frame matching via MASt3R.

    Loads the official MASt3R model and provides a ``match_pair`` method
    that returns filtered 3D-3D correspondences.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda:0",
        min_confidence: float = 0.5,
    ) -> None:
        self.checkpoint = _resolve_checkpoint(checkpoint or MAST3R_CHECKPOINT_DEFAULT)
        self.device = device
        self.min_confidence = min_confidence
        self._model = None  # lazy-load

    @property
    def version(self) -> str:
        return f"MASt3R@{MAST3R_COMMIT}:{self.checkpoint}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_pair(
        self,
        frames_i: list[str],
        frames_j: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match overlapping frames between two segments.

        Parameters
        ----------
        frames_i : list[str]
            Frame paths from segment *i*.
        frames_j : list[str]
            Frame paths from segment *j*.

        Returns
        -------
        pts_i : np.ndarray  (N, 3)
            3D points in segment *i*'s coordinate frame.
        pts_j : np.ndarray  (N, 3)
            Corresponding 3D points in segment *j*'s coordinate frame.
        confidences : np.ndarray  (N,)
            Per-match confidence scores.

        Raises
        ------
        InsufficientMatchesError
            If fewer than 3 non-collinear correspondences survive confidence
            filtering.
        """
        _setup_mast3r_path()

        import torch
        from PIL import Image
        from PIL.ImageOps import exif_transpose

        import mast3r.utils.path_to_dust3r  # noqa: F401  # adds dust3r to sys.path
        import dust3r.utils.path_to_croco  # noqa: F401
        from dust3r.utils.image import ImgNorm, _resize_pil_image
        from dust3r.inference import inference
        from dust3r.utils.device import to_numpy
        from mast3r.fast_nn import extract_correspondences_nonsym

        # Lazy-load model
        if self._model is None:
            from mast3r.model import AsymmetricMASt3R

            self._model = AsymmetricMASt3R.from_pretrained(self.checkpoint).to(self.device)
            self._model.eval()

        all_pts_i: list[np.ndarray] = []
        all_pts_j: list[np.ndarray] = []
        all_confs: list[np.ndarray] = []

        for fp_i, fp_j in zip(frames_i, frames_j):
            # Load and format images in the DUSt3R convention
            img_i = _load_dust3r_image(fp_i)
            img_j = _load_dust3r_image(fp_j)

            # Run inference on this pair
            res = inference([(img_i, img_j)], self._model, self.device, batch_size=1, verbose=False)
            pred1 = res["pred1"][0]  # first batch, view 1
            pred2 = res["pred2"][0]  # first batch, view 2

            # Match descriptors with reciprocal nearest neighbours
            corres = extract_correspondences_nonsym(
                pred1["desc"], pred2["desc"],
                pred1["conf"], pred2["conf"],
                device=self.device,
            )
            xy1, xy2, conf = to_numpy(corres)

            # Confidence filter
            mask = conf >= self.min_confidence
            xy1, xy2, conf = xy1[mask], xy2[mask], conf[mask]

            if len(conf) == 0:
                continue

            # Look up 3D points at matched pixel positions
            pts3d_1 = to_numpy(pred1["pts3d"])
            pts3d_2 = to_numpy(pred2["pts3d"])
            all_pts_i.append(pts3d_1[xy1[:, 1], xy1[:, 0]])
            all_pts_j.append(pts3d_2[xy2[:, 1], xy2[:, 0]])
            all_confs.append(conf)

        if not all_pts_i:
            raise InsufficientMatchesError("overlap", 0, 3)

        pts_i = np.concatenate(all_pts_i, axis=0)
        pts_j = np.concatenate(all_pts_j, axis=0)
        confs = np.concatenate(all_confs, axis=0)

        if len(pts_i) < 3:
            raise InsufficientMatchesError("overlap", len(pts_i), 3)

        return pts_i.astype(np.float32), pts_j.astype(np.float32), confs.astype(np.float32)


def _load_dust3r_image(path: str, size: int = 512) -> dict:
    """Load a single image in the format expected by DUSt3R/MASt3R inference."""
    from PIL import Image
    from PIL.ImageOps import exif_transpose
    import numpy as np
    from dust3r.utils.image import ImgNorm, _resize_pil_image

    img = exif_transpose(Image.open(path)).convert("RGB")
    img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        idx=0,
        instance="0",
    )


# ---------------------------------------------------------------------------
# Sim(3) estimation (RANSAC)
# ---------------------------------------------------------------------------


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama similarity transform (scale, 3x3 rotation, 3-translation)."""
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = np.linalg.norm(src_c) / np.sqrt(n)
    sigma_dst = np.linalg.norm(dst_c) / np.sqrt(n)

    H = (dst_c.T @ src_c) / n
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    scale = (sigma_dst / sigma_src) if sigma_src > 1e-10 else 1.0
    t = mu_dst - scale * (R @ mu_src)
    return float(scale), R.astype(np.float32), t.astype(np.float32)


def _is_collinear(pts: np.ndarray, eps: float = 1e-8) -> bool:
    """Return True if *pts* (N,3) are (nearly) collinear."""
    if pts.shape[0] < 3:
        return True
    centered = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)
    return bool(s[1] < eps)


def estimate_sim3_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    thresh: float = 0.05,
    max_iters: int = 1000,
    min_inliers: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """RANSAC-robust Sim(3) estimation.

    Parameters
    ----------
    src, dst : (N, 3) arrays of corresponding points.
    thresh : Inlier distance threshold.
    max_iters : Maximum RANSAC iterations.
    min_inliers : Minimum required inliers.

    Returns
    -------
    scale, R, t, num_inliers

    Raises
    ------
    InsufficientMatchesError
        If fewer than 3 (non-collinear) correspondences exist, or if the
        final inlier count is below *min_inliers*.
    """
    n = src.shape[0]
    if n < 3:
        raise InsufficientMatchesError("(unknown)", n, 3)

    best_inliers = 0
    best_model = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    for _ in range(max_iters):
        idx = np.random.choice(n, 3, replace=False)
        if _is_collinear(src[idx]):
            continue

        scale, R, t = _umeyama(src[idx], dst[idx])
        transformed = scale * (R @ src.T).T + t
        errors = np.linalg.norm(transformed - dst, axis=1)
        inliers = int(np.sum(errors < thresh))

        if inliers > best_inliers:
            best_inliers = inliers
            best_model = (scale, R, t)
            if inliers > n * 0.9:
                break

    inlier_mask = (
        np.linalg.norm(best_model[0] * (best_model[1] @ src.T).T + best_model[2] - dst, axis=1)
        < thresh
    )

    if int(np.sum(inlier_mask)) >= 3:
        scale, R, t = _umeyama(src[inlier_mask], dst[inlier_mask])
        best_model = (scale, R.astype(np.float32), t.astype(np.float32))

    final_inliers = int(np.sum(inlier_mask))
    if final_inliers < min_inliers:
        raise InsufficientMatchesError("(ransac)", final_inliers, min_inliers)

    return (*best_model, final_inliers)
