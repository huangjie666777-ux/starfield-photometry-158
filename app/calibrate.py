"""Per-frame differential photometric calibration (zero-point estimation).

For each valid reference star k:
    ZP_k = mag_k + 2.5 log10(flux_rate_k)
with propagated magnitude uncertainty
    sig_ZPk^2 = mag_err_k^2 + (1.08574 * ferr/f)^2.
A MAD-based robust pass rejects reference-star outliers (intrinsic variability,
crowding residuals, cosmic rays); the target never enters the calibration, so
frame-to-frame transparency changes are removed while target variability is
preserved.  At least three surviving references are required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_REFS = 3


@dataclass
class RefMeasurement:
    id: str
    mag: float
    mag_err: float
    flux_rate: float | None = None
    flux_rate_err: float | None = None
    valid: bool = True
    reject_reason: str | None = None


@dataclass
class CalibrationResult:
    ok: bool
    reason: str | None = None
    zero_point: float | None = None
    zero_point_err: float | None = None
    refs: list[RefMeasurement] = field(default_factory=list)


def calibrate_zero_point(refs: list[RefMeasurement]) -> CalibrationResult:
    res = CalibrationResult(ok=False, refs=refs)
    zps, sigs, idx = [], [], []
    for i, r in enumerate(refs):
        if not r.valid or r.flux_rate is None or r.flux_rate_err is None:
            continue
        if r.flux_rate <= 0:
            r.valid = False
            r.reject_reason = "non-positive flux"
            continue
        zp = r.mag + 2.5 * np.log10(r.flux_rate)
        sig = np.sqrt(r.mag_err ** 2
                      + (1.08574 * r.flux_rate_err / r.flux_rate) ** 2)
        zps.append(zp)
        sigs.append(sig)
        idx.append(i)

    if len(zps) < MIN_REFS:
        res.reason = (f"only {len(zps)} valid reference stars; need >= {MIN_REFS}")
        return res

    zps = np.asarray(zps)
    keep = np.ones(len(zps), dtype=bool)
    # Two-pass MAD rejection: a 4-MAD gate, tightened to 3-MAD.
    for gate in (4.0, 3.0):
        z = zps[keep]
        med = float(np.median(z))
        mad = 1.4826 * float(np.median(np.abs(z - med)))
        if not np.isfinite(mad) or mad <= 0:
            break
        keep = np.abs(zps - med) < gate * max(mad, 1e-6)
        if keep.sum() < MIN_REFS:
            keep[:] = False
            # Restore the last acceptable set by widening to 4 MAD.
            med0 = float(np.median(zps))
            mad0 = 1.4826 * float(np.median(np.abs(zps - med0)))
            keep = np.abs(zps - med0) < 4.0 * max(mad0, 1e-6)
            break

    for j, i in enumerate(idx):
        if not keep[j]:
            refs[i].valid = False
            refs[i].reject_reason = "zero-point outlier"

    if keep.sum() < MIN_REFS:
        res.reason = (f"only {int(keep.sum())} references survive outlier "
                      f"rejection; need >= {MIN_REFS}")
        return res

    z = zps[keep]
    s = np.asarray(sigs)[keep]
    zp_mean = float(np.average(z, weights=1.0 / np.maximum(s, 1e-9) ** 2))
    # Zero-point uncertainty: propagated per-ref errors plus frame scatter.
    zp_scatter = float(np.std(z, ddof=1)) if len(z) > 1 else 0.0
    mean_stat = float(np.sqrt(1.0 / np.sum(1.0 / np.maximum(s, 1e-9) ** 2)))
    res.ok = True
    res.zero_point = zp_mean
    res.zero_point_err = float(np.hypot(zp_scatter / np.sqrt(len(z)), mean_stat))
    return res


def target_magnitude(flux_rate: float, flux_rate_err: float,
                     zp: float, zp_err: float):
    """mag = ZP - 2.5 log10(flux/exptime-normalized rate), propagated errors."""
    mag = zp - 2.5 * np.log10(flux_rate)
    mag_err = np.sqrt(zp_err ** 2
                      + (1.08574 * flux_rate_err / flux_rate) ** 2)
    return float(mag), float(mag_err)
