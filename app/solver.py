"""Shared plate-solving pipeline used by both /api/solve and photometry.

Reuses source extraction (extract.py), geometric matching (match.py), the
robust affine fit (fit.py) and WCS construction (wcsbuild.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import ExtractionError, extract_sources
from .fit import FitError, robust_fit
from .match import MatchError, assign_pairs, match, project_tangent
from .wcsbuild import affine_to_wcs_header


class SolveError(ValueError):
    """Plate solving failed for this frame."""


@dataclass
class PairRecord:
    id: str
    x: float
    y: float
    residual_arcsec: float


@dataclass
class SolveResult:
    A: np.ndarray          # pixel = A @ tangent_arcsec + b
    b: np.ndarray
    rms_arcsec: float
    mirrored: bool
    wcs: dict
    pairs: list  # list[PairRecord]


def solve_frame(data: np.ndarray, catalog, center_ra: float, center_dec: float,
                pixel_scale_min: float, pixel_scale_max: float, rms_max: float,
                threshold_sigma: float = 5.0, saturation: float | None = None,
                max_pairs: int = 200) -> SolveResult:
    """Run the full extraction -> match -> fit -> WCS pipeline on one frame."""
    try:
        sources, _, _ = extract_sources(
            data, threshold_sigma=threshold_sigma, saturation=saturation)
    except ExtractionError as exc:
        raise SolveError(f"extraction failed: {exc}") from exc
    if len(sources) < 6:
        raise SolveError(f"only {len(sources)} usable sources detected; need >= 6")

    ra = [s.ra for s in catalog]
    dec = [s.dec for s in catalog]
    try:
        xi, eta = project_tangent(ra, dec, center_ra, center_dec)
    except MatchError as exc:
        raise SolveError(str(exc)) from exc
    cat_xy = np.column_stack([xi, eta])

    try:
        src_xy = np.array([[s.x, s.y] for s in sources])
        pairs, _, _ = match(src_xy, cat_xy,
                            1.0 / pixel_scale_max, 1.0 / pixel_scale_min)
        A, b, kept, resid, rms = robust_fit(src_xy, cat_xy, pairs,
                                            rms_max, max_pairs)
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        pairs2 = assign_pairs(src_xy, cat_xy, A, b, 3.0 * scale)
        if len(pairs2) > len(kept):
            A, b, kept, _, rms = robust_fit(src_xy, cat_xy, pairs2,
                                            rms_max, max_pairs)
        # Residuals of the final kept pairs for reporting.
        kept_idx = np.array([[s, c] for s, c in kept])
        pred = cat_xy[kept_idx[:, 1]] @ A.T + b
        d = np.hypot(*(pred - src_xy[kept_idx[:, 0]]).T)
    except (MatchError, FitError) as exc:
        raise SolveError(str(exc)) from exc

    wcs = affine_to_wcs_header(A, b, center_ra, center_dec)
    pair_records = [
        PairRecord(id=catalog[int(ci)].id, x=float(src_xy[int(si), 0]),
                   y=float(src_xy[int(si), 1]), residual_arcsec=float(r))
        for (si, ci), r in zip(kept, d)
    ]
    return SolveResult(A=A, b=b, rms_arcsec=float(rms),
                       mirrored=bool(np.linalg.det(A) < 0), wcs=wcs,
                       pairs=pair_records)


def catalog_pixel_positions(sol: SolveResult, catalog,
                            center_ra: float, center_dec: float) -> np.ndarray:
    """Project every catalog star to pixel coordinates on the solved frame."""
    xi, eta = project_tangent([s.ra for s in catalog], [s.dec for s in catalog],
                              center_ra, center_dec)
    tangent = np.column_stack([xi, eta])
    return tangent @ sol.A.T + sol.b
