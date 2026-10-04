"""Multi-frame differential light-curve assembly.

Each frame reuses the same plate-solving pipeline, catalog stars are
projected to pixels, then aperture photometry and the per-frame zero point
are measured.  A single bad frame never aborts the run: it is recorded with
its failure reason while the remaining frames continue.  Records are sorted
by UTC exposure midpoint (MJD).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .aperture import aperture_photometry
from .calibrate import (CalibrationResult, RefMeasurement, calibrate_zero_point,
                        target_magnitude)
from .solver import SolveError, catalog_pixel_positions, solve_frame


@dataclass
class FrameRecord:
    frame: int
    ok: bool
    reason: str | None
    mjd_mid: float | None
    flux_rate: float | None
    flux_rate_err: float | None
    target_mag: float | None
    target_mag_err: float | None
    zero_point: float | None
    zero_point_err: float | None
    refs: list[RefMeasurement]


def exposure_mid_mjd(hdr: fits.Header) -> tuple[float | None, float | None,
                                                str | None]:
    """Return (mid_mjd, exptime, error). DATE-OBS is the exposure start UTC."""
    if "EXPTIME" not in hdr:
        return None, None, "missing EXPTIME"
    try:
        exptime = float(hdr["EXPTIME"])
    except (TypeError, ValueError):
        return None, None, "non-numeric EXPTIME"
    if not np.isfinite(exptime) or exptime <= 0:
        return None, None, "EXPTIME must be positive"

    dateobs = hdr.get("DATE-OBS", hdr.get("DATEOBS"))
    if dateobs is None and "MJD-OBS" in hdr:
        try:
            start_mjd = float(hdr["MJD-OBS"])
        except (TypeError, ValueError):
            return None, None, "invalid MJD-OBS"
    else:
        if dateobs is None:
            return None, None, "missing DATE-OBS"
        try:
            t = Time(str(dateobs).strip(), format="isot", scale="utc",
                     precision=9)
            start_mjd = float(t.mjd)
        except (ValueError, TypeError):
            try:  # bare date YYYY-MM-DD
                t = Time(str(dateobs).strip(), format="iso", scale="utc")
                start_mjd = float(t.mjd)
            except (ValueError, TypeError):
                return None, None, f"invalid DATE-OBS: {dateobs!r}"
    if not np.isfinite(start_mjd):
        return None, None, "non-finite exposure time"
    return start_mjd + exptime / 2.0 / 86400.0, exptime, None


def process_frame(frame_no: int, hdu: fits.PrimaryHDU, p) -> FrameRecord:
    """Solve, measure and calibrate one frame; always returns a record."""
    hdr = hdu.header
    mjd, exptime, err = exposure_mid_mjd(hdr)
    if err:
        return FrameRecord(frame_no, False, err, None, None, None, None, None,
                           None, None, [])

    try:
        data = np.asarray(hdu.data, dtype=np.float64)
    except Exception as exc:
        return FrameRecord(frame_no, False, f"unreadable image: {exc}", mjd,
                           None, None, None, None, None, None, [])
    if data.ndim != 2 or not np.all(np.isfinite(data.shape)):
        return FrameRecord(frame_no, False, "image is not 2D", mjd, None, None,
                           None, None, None, None, [])

    # Reuse the exact plate-solving pipeline for every frame.
    try:
        sol = solve_frame(data, p.catalog, p.center_ra, p.center_dec,
                          p.pixel_scale_min, p.pixel_scale_max, p.rms_max,
                          p.threshold_sigma, p.saturation, p.max_pairs)
        xy = catalog_pixel_positions(sol, p.catalog, p.center_ra, p.center_dec)
    except SolveError as exc:
        return FrameRecord(frame_no, False, f"plate solve failed: {exc}", mjd,
                           None, None, None, None, None, None, [])

    id_index = {s.id: i for i, s in enumerate(p.catalog)}
    ti = id_index[p.target_id]
    ref_ids = [r.id for r in p.references]
    ref_idx = [id_index[i] for i in ref_ids]

    common = dict(aperture_radius=p.aperture_radius,
                  inner_radius=p.inner_radius, outer_radius=p.outer_radius,
                  gain=p.gain, read_noise=p.read_noise, exptime=exptime,
                  saturation=p.saturation)

    def measure(i: int):
        others = np.delete(xy, i, axis=0)
        return aperture_photometry(data, float(xy[i, 0]), float(xy[i, 1]),
                                   others, **common)

    ref_records: list[RefMeasurement] = []
    for spec, ri in zip(p.references, ref_idx):
        rm = RefMeasurement(id=spec.id, mag=spec.mag, mag_err=spec.mag_err)
        ar = measure(ri)
        if ar.ok:
            rm.flux_rate = ar.flux_rate
            rm.flux_rate_err = ar.flux_rate_err
        else:
            rm.valid = False
            rm.reject_reason = ar.reason
        ref_records.append(rm)

    cal: CalibrationResult = calibrate_zero_point(ref_records)
    if not cal.ok:
        return FrameRecord(frame_no, False, cal.reason, mjd, None, None, None,
                           None, None, None, ref_records)

    tar = measure(ti)
    if not tar.ok:
        return FrameRecord(frame_no, False, f"target: {tar.reason}", mjd, None,
                           None, None, None, cal.zero_point, cal.zero_point_err,
                           ref_records)

    mag, mag_err = target_magnitude(tar.flux_rate, tar.flux_rate_err,
                                    cal.zero_point, cal.zero_point_err)
    return FrameRecord(frame_no, True, None, mjd, tar.flux_rate,
                       tar.flux_rate_err, mag, mag_err, cal.zero_point,
                       cal.zero_point_err, ref_records)
