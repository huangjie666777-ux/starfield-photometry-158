"""FastAPI entry point: /api/solve and solved-FITS download (requirements 1, 5, 6)."""
from __future__ import annotations

import csv
import io
import uuid

import numpy as np
from astropy.io import fits
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .models import (MAX_FITS_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE,
                     MAX_FRAMES, FrameResult, PairOut, PhotoParams,
                     PhotoResponse, RefOut, SolveParams, SolveResponse)
from .solver import SolveError, solve_frame
from .wcsbuild import solved_fits_bytes

from .lightcurve import process_frame

app = FastAPI(title="Star Field Plate Solver")

# In-memory store of solved FITS products, keyed by solve_id.
_RESULTS: dict[str, bytes] = {}
_PHOTO_CSV: dict[str, bytes] = {}
_PHOTO_RESULT: dict[str, PhotoResponse] = {}


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


@app.post("/api/solve", response_model=SolveResponse)
async def solve(image: UploadFile = File(...), params: str = Form(...)):
    # ---- Parse and validate parameters (requirement 1) ----
    try:
        p = SolveParams.model_validate_json(params)
    except Exception as exc:
        _err(422, f"invalid parameters: {exc}")

    blob = await image.read()
    if len(blob) > MAX_FITS_BYTES:
        _err(413, "FITS file too large")
    try:
        hdul = fits.open(io.BytesIO(blob), memmap=False)
        hdu = hdul[0]
        if not isinstance(hdu, fits.PrimaryHDU) or hdu.data is None:
            _err(422, "FITS must contain a primary HDU with image data")
        data = np.asarray(hdu.data, dtype=np.float64)
    except HTTPException:
        raise
    except Exception:
        _err(422, "could not read FITS file")
    if data.ndim != 2:
        _err(422, "primary HDU data must be a 2D image")
    if data.size > MAX_IMAGE_PIXELS or max(data.shape) > MAX_IMAGE_SIDE:
        _err(413, f"image exceeds limit of {MAX_IMAGE_SIDE}px side / "
                  f"{MAX_IMAGE_PIXELS} pixels")

    # ---- Reuse the shared solve pipeline ----
    try:
        sol = solve_frame(data, p.catalog, p.center_ra, p.center_dec,
                          p.pixel_scale_min, p.pixel_scale_max, p.rms_max,
                          p.threshold_sigma, p.saturation, p.max_pairs)
    except SolveError as exc:
        _err(422, str(exc))

    out_pairs = [PairOut(id=q.id, x=q.x, y=q.y,
                         residual_arcsec=q.residual_arcsec)
                 for q in sol.pairs]

    solve_id = uuid.uuid4().hex
    _RESULTS[solve_id] = solved_fits_bytes(hdu, sol.wcs)
    hdul.close()
    return SolveResponse(solve_id=solve_id, n_pairs=len(out_pairs),
                         rms_arcsec=sol.rms_arcsec, mirrored=sol.mirrored,
                         pairs=out_pairs, wcs=sol.wcs)


@app.get("/api/solve/{solve_id}/fits")
def download_fits(solve_id: str):
    blob = _RESULTS.get(solve_id)
    if blob is None:
        _err(404, "unknown solve_id")
    return Response(content=blob, media_type="application/fits",
                    headers={"Content-Disposition":
                             "attachment; filename=solved.fits"})


def _load_image_hdu(blob: bytes) -> fits.PrimaryHDU:
    try:
        hdul = fits.open(io.BytesIO(blob), memmap=False)
        hdu = hdul[0]
    except Exception:
        _err(422, "could not read FITS file")
    if not isinstance(hdu, fits.PrimaryHDU) or hdu.data is None:
        _err(422, "FITS must contain a primary HDU with image data")
    data = np.asarray(hdu.data, dtype=np.float64)
    if data.ndim != 2:
        _err(422, "primary HDU data must be a 2D image")
    if data.size > MAX_IMAGE_PIXELS or max(data.shape) > MAX_IMAGE_SIDE:
        _err(413, f"image exceeds limit of {MAX_IMAGE_SIDE}px side / "
                  f"{MAX_IMAGE_PIXELS} pixels")
    return hdu


def _try_load_hdu(blob: bytes):
    try:
        return _load_image_hdu(blob), None
    except HTTPException as exc:
        return None, str(exc.detail)


@app.post("/api/photometry", response_model=PhotoResponse)
async def photometry(images: list[UploadFile] = File(...),
                     params: str = Form(...)):
    """Differential aperture photometry over 2-20 same-filter calibrated frames."""
    if not (2 <= len(images) <= MAX_FRAMES):
        _err(422, f"expected between 2 and {MAX_FRAMES} FITS frames, "
                  f"got {len(images)}")
    try:
        p = PhotoParams.model_validate_json(params)
    except Exception as exc:
        _err(422, f"invalid parameters: {exc}")

    blobs = []
    for up in images:
        blob = await up.read()
        if len(blob) > MAX_FITS_BYTES:
            _err(413, "FITS file too large")
        blobs.append(blob)

    # Frames are processed independently: one failure is recorded, others run.
    records = []
    for i, blob in enumerate(blobs, start=1):
        hdu, load_err = _try_load_hdu(blob)
        if load_err is not None:
            records.append(FrameResult(frame=i, ok=False, reason=load_err))
            continue
        try:
            records.append(process_frame(i, hdu, p))
        except Exception as exc:  # never let one frame kill the light curve
            records.append(FrameResult(frame=i, ok=False,
                                       reason=f"unexpected error: {exc}"))

    # Stable sort by exposure midpoint; frames without a valid time go last.
    records.sort(key=lambda r: (r.mjd_mid is None,
                                r.mjd_mid if r.mjd_mid is not None else 0.0,
                                r.frame))

    out_frames = []
    for rec in records:
        refs = [RefOut(id=q.id, used=q.valid, reason=q.reject_reason)
                for q in rec.refs]
        out_frames.append(FrameResult(
            frame=rec.frame, ok=rec.ok, reason=rec.reason,
            mjd_mid=rec.mjd_mid, flux_rate=rec.flux_rate,
            flux_rate_err=rec.flux_rate_err, target_mag=rec.target_mag,
            target_mag_err=rec.target_mag_err, zero_point=rec.zero_point,
            zero_point_err=rec.zero_point_err,
            n_refs_used=sum(1 for q in rec.refs if q.valid), references=refs))

    resp = PhotoResponse(target_id=p.target_id, n_frames=len(out_frames),
                         n_ok=sum(1 for f in out_frames if f.ok),
                         frames=out_frames)
    pid = uuid.uuid4().hex
    _PHOTO_RESULT[pid] = resp
    _PHOTO_CSV[pid] = _photometry_csv(resp)
    # Bound memory usage of in-memory products.
    _evict_old(_PHOTO_CSV)
    _evict_old(_PHOTO_RESULT)
    from fastapi.responses import JSONResponse
    body = resp.model_dump()
    body["photometry_id"] = pid
    return JSONResponse(body)


def _evict_old(store: dict, keep: int = 200):
    while len(store) > keep:
        store.pop(next(iter(store)))


CSV_COLUMNS = ["sort_index", "frame", "ok", "reason", "mjd_mid",
               "flux_rate_e_per_s", "flux_rate_err_e_per_s", "target_mag",
               "target_mag_err", "zero_point", "zero_point_err",
               "n_refs_used", "refs_used", "refs_excluded"]


def _fmt(v):
    return "" if v is None else (f"{v:.10g}" if isinstance(v, float) else str(v))


def _photometry_csv(resp: PhotoResponse) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for sort_i, f in enumerate(resp.frames):
        used = ";".join(r.id for r in f.references if r.used)
        excluded = ";".join(f"{r.id}:{r.reason or ''}" for r in f.references
                             if not r.used)
        w.writerow([sort_i, f.frame, f.ok, f.reason or "", _fmt(f.mjd_mid),
                    _fmt(f.flux_rate), _fmt(f.flux_rate_err),
                    _fmt(f.target_mag), _fmt(f.target_mag_err),
                    _fmt(f.zero_point), _fmt(f.zero_point_err),
                    f.n_refs_used, used, excluded])
    return buf.getvalue().encode("utf-8")


@app.get("/api/photometry/{photometry_id}/csv")
def download_photometry_csv(photometry_id: str):
    blob = _PHOTO_CSV.get(photometry_id)
    if blob is None:
        _err(404, "unknown photometry_id")
    return Response(content=blob, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition":
                             "attachment; filename=photometry.csv"})
