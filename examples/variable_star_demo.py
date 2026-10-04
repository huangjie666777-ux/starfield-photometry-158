#!/usr/bin/env python3
"""End-to-end variable-star demonstration.

Generates 8 calibrated, same-filter synthetic FITS frames in which:
  * the target varies sinusoidally (+-0.35 mag);
  * the sky transparency changes frame to frame (common-mode extinction);
  * the reference stars are constant.
Posts them to /api/photometry, downloads the CSV, and prints the recovered
differential light curve plus the per-frame zero points.

Usage:
    .venv/bin/python examples/variable_star_demo.py --port 8152
The server must already be running (see README).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.match import project_tangent

RA0, DEC0, SCALE = 150.0, 20.0, 1.3
NY, NX, GAIN, RN, ZP0 = 360, 360, 1.8, 7.0, 25.0
REF_IDS = [f"star{i:03d}" for i in (1, 5, 10, 15, 20, 25)]


def make_catalog(n=30, seed=123):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(60)))
    gx, gy = np.meshgrid(np.linspace(-0.045, 0.045, side),
                         np.linspace(-0.045, 0.045, side))
    gx = (gx.ravel() + rng.uniform(-0.008, 0.008, gx.size)
          / np.cos(np.deg2rad(DEC0)))
    gy = gy.ravel() + rng.uniform(-0.008, 0.008, gy.size)
    ras = RA0 + gx / np.cos(np.deg2rad(DEC0))
    decs = DEC0 + gy
    return [f"star{k:03d}" for k in range(n)], ras[:n], decs[:n]


IDS, RAS, DECS = make_catalog()


def render(target_rate, transp, dateobs, seed):
    rng = np.random.default_rng(seed)
    xi, eta = project_tangent(RAS, DECS, RA0, DEC0)
    th = np.deg2rad(20.0)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) / SCALE
    pix = np.column_stack([xi, eta]) @ Rm.T + [NX / 2, NY / 2]
    sky = 500.0
    img = rng.normal(sky, np.sqrt(sky * GAIN + RN ** 2) / GAIN, (NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    rr = np.random.default_rng(99)
    ref_mag = {}
    for i, sid in enumerate(IDS):
        if sid == "star000":
            rate = target_rate
        elif sid in REF_IDS:
            mag = rr.uniform(12.0, 14.5)
            ref_mag[sid] = mag
            rate = 10 ** ((ZP0 - mag) / 2.5)
        else:
            rate = 10 ** ((ZP0 - 14.5) / 2.5)
        img += (rate * 30 * transp / GAIN) * np.exp(
            -((xx - pix[i, 0]) ** 2 + (yy - pix[i, 1]) ** 2) / (2 * 1.6 ** 2))
    hdu = fits.PrimaryHDU(img.astype(np.float32))
    hdu.header["DATE-OBS"] = dateobs
    hdu.header["EXPTIME"] = 30.0
    hdu.header["FILTER"] = "V"
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue(), ref_mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8152)
    args = ap.parse_args()
    transps = [1.0, 0.72, 0.88, 0.6, 0.95, 0.78, 1.0, 0.68]
    files, ref_mags = [], None
    for k, (tr, ph) in enumerate(zip(transps, np.linspace(0, 2 * np.pi, 8,
                                                         endpoint=False))):
        rate = 10 ** ((ZP0 - (13.0 + 0.35 * np.sin(ph))) / 2.5)
        blob, ref_mag = render(rate, tr, f"2025-03-10T01:{k*2:02d}:00", k)
        files.append(("images", (f"f{k}.fits", blob, "application/fits")))
        ref_mags = ref_mag
    catalog = [{"id": i, "ra": float(r), "dec": float(d)}
               for i, r, d in zip(IDS, RAS, DECS)]
    params = {
        "catalog": catalog, "center_ra": RA0, "center_dec": DEC0,
        "pixel_scale_min": 1.0, "pixel_scale_max": 1.6, "rms_max": 0.5,
        "target_id": "star000",
        "references": [{"id": i, "mag": float(m), "mag_err": 0.01}
                       for i, m in ref_mags.items()],
        "aperture_radius": 4.0, "inner_radius": 7.0, "outer_radius": 12.0,
        "gain": GAIN, "read_noise": RN,
    }
    url = f"http://127.0.0.1:{args.port}"
    r = httpx.post(f"{url}/api/photometry", files=files,
                   data={"params": json.dumps(params)}, timeout=120)
    r.raise_for_status()
    body = r.json()
    print(f"{'MJD':>14} {'flux/s(e)':>12} {'mag':>8} {'mag_err':>8} "
          f"{'ZP':>8} {'ok':>5}")
    for f in body["frames"]:
        print(f"{f['mjd_mid']:14.5f} {f['flux_rate'] or float('nan'):12.1f} "
              f"{f['target_mag'] or float('nan'):8.4f} "
              f"{f['target_mag_err'] or float('nan'):8.4f} "
              f"{f['zero_point'] or float('nan'):8.4f} {str(f['ok']):>5}")
    csv_blob = httpx.get(
        f"{url}/api/photometry/{body['photometry_id']}/csv").content
    with open("photometry_demo.csv", "wb") as fh:
        fh.write(csv_blob)
    print("wrote photometry_demo.csv")


if __name__ == "__main__":
    main()
