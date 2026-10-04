"""Differential photometry tests on synthetic calibrated light-curve frames."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app.main import app
from app.match import project_tangent

CENTER_RA, CENTER_DEC = 150.0, 20.0
SCALE = 1.3
NY, NX = 360, 360
GAIN, RN = 1.8, 7.0
ZP0 = 25.0  # mag = ZP0 - 2.5 log10(flux_rate_e)


def _grid_catalog(n=36):
    rng = np.random.default_rng(123)
    # Deterministic jittered layout: regular grids have symmetric triangles
    # that make geometric matching ambiguous.  The first *n* positions are
    # bright in-FOV stars; the rest are dim and outside the rendered FOV.
    side = int(np.ceil(np.sqrt(60)))
    gx, gy = np.meshgrid(np.linspace(-0.045, 0.045, side),
                         np.linspace(-0.045, 0.045, side))
    gx = (gx.ravel() + rng.uniform(-0.008, 0.008, gx.size)
          / np.cos(np.deg2rad(CENTER_DEC)))
    gy = gy.ravel() + rng.uniform(-0.008, 0.008, gy.size)
    ras = CENTER_RA + gx / np.cos(np.deg2rad(CENTER_DEC))
    decs = CENTER_DEC + gy
    ids = [f"star{k:03d}" for k in range(len(ras))]
    return ids[:n], ras[:n], decs[:n]


IDS, RAS, DECS = _grid_catalog(30)


def render_frame(target_rate, transparency, exptime=30.0, dateobs=None,
                 seed=0, bad_target_pixel=False):
    rng = np.random.default_rng(seed)
    xi, eta = project_tangent(RAS, DECS, CENTER_RA, CENTER_DEC)
    th = np.deg2rad(20.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = R / SCALE
    t = np.array([NX / 2.0, NY / 2.0])
    pix = np.column_stack([xi, eta]) @ A.T + t

    sky_adu = 500.0
    img = rng.normal(sky_adu, np.sqrt(sky_adu * GAIN + RN ** 2) / GAIN,
                     (NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    rates = {}
    # Assign rates: star000 is the variable target; six fixed references.
    ref_ids = [f"star{i:03d}" for i in (1, 5, 10, 15, 20, 25)]
    rng2 = np.random.default_rng(42)
    for rid in ref_ids:
        rates[rid] = 10 ** ((ZP0 - rng2.uniform(12.0, 14.5)) / 2.5)
    rates["star000"] = target_rate
    for i, sid in enumerate(IDS):
        if sid in ref_ids or sid == "star000":
            rate = rates[sid]
        else:
            # Non-reference field stars: moderately bright, stable.
            rate = 10 ** ((ZP0 - 14.5) / 2.5)
        electrons = rate * exptime * transparency
        amp_adu = electrons / GAIN
        img += amp_adu * np.exp(-((xx - pix[i, 0]) ** 2
                                  + (yy - pix[i, 1]) ** 2) / (2 * 1.6 ** 2))
    if bad_target_pixel:
        ci = int(round(pix[0, 0]))
        img[int(round(pix[0, 1])), ci] = np.nan

    hdu = fits.PrimaryHDU(data=img.astype(np.float32))
    hdu.header["TELESCOP"] = "SYNTH"
    hdu.header["PC_TEMP"] = 12.3   # observer keyword starting with "PC"
    hdu.header["EXPTIME"] = exptime
    hdu.header["DATE-OBS"] = dateobs
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue(), ref_ids, rates


def photo_params(ref_ids, rates, refs_mag, **over):
    catalog = [{"id": i, "ra": float(r), "dec": float(d)}
               for i, r, d in zip(IDS, RAS, DECS)]
    p = {
        "catalog": catalog,
        "center_ra": CENTER_RA, "center_dec": CENTER_DEC,
        "pixel_scale_min": 1.0, "pixel_scale_max": 1.6,
        "rms_max": 0.5,
        "target_id": "star000",
        "references": [{"id": rid, "mag": refs_mag[rid], "mag_err": 0.01}
                       for rid in ref_ids],
        "aperture_radius": 4.0, "inner_radius": 7.0, "outer_radius": 12.0,
        "gain": GAIN, "read_noise": RN,
    }
    p.update(over)
    return p


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_variable_target_recovered_under_transparency_changes(client):
    # Target varies +/-0.3 mag; transparency swings 60%..100% (common mode).
    phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    files, meta = [], None
    for k, ph in enumerate(phases):
        target_rate = 10 ** ((ZP0 - (13.0 + 0.3 * np.sin(ph))) / 2.5)
        transp = [1.0, 0.7, 0.85, 0.6, 0.95, 0.75][k]
        blob, ref_ids, rates = render_frame(
            target_rate, transp, dateobs=f"2025-03-01T00:{10 + k:02d}:00",
            seed=k)
        files.append((f"f{k}.fits", blob))
        meta = (ref_ids, rates, target_rate)
    ref_ids, rates, _ = meta
    refs_mag = {rid: ZP0 - 2.5 * np.log10(rate) for rid, rate in rates.items()
                if rid in ref_ids}
    params = photo_params(ref_ids, rates, refs_mag)
    r = client.post("/api/photometry",
                    files=[("images", (n, b, "application/fits")) for n, b in files],
                    data={"params": json.dumps(params)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_frames"] == 6 and body["n_ok"] == 6
    frames = body["frames"]
    # Sorted by MJD ascending.
    mjds = [f["mjd_mid"] for f in frames]
    assert mjds == sorted(mjds)
    mags = np.array([f["target_mag"] for f in frames])
    # Zero points track transparency: fainter sky -> smaller ZP.
    zps = np.array([f["zero_point"] for f in frames])
    assert zps[3] < zps[0] - 0.4  # transp 0.6 vs 1.0
    # Recovered sinusoidal modulation (differential calibration removes it).
    recovered = 0.5 * (mags.max() - mags.min())
    assert recovered == pytest.approx(0.3, abs=0.08)
    assert np.all(np.array([f["n_refs_used"] for f in frames]) >= 3)
    # CSV download is consistent with JSON.
    r2 = client.get(f"/api/photometry/{body['photometry_id']}/csv")
    assert r2.status_code == 200
    text_csv = r2.content.decode()
    assert "mjd_mid" in text_csv and "target_mag" in text_csv
    assert text_csv.count("\n") == 7


def test_failed_frame_is_recorded_but_others_continue(client):
    good, ref_ids, rates = render_frame(10 ** ((ZP0 - 13.0) / 2.5), 1.0,
                                        dateobs="2025-03-02T00:00:00")
    # Second frame: NaN inside target aperture -> target failure recorded.
    bad, _, _ = render_frame(10 ** ((ZP0 - 13.0) / 2.5), 1.0,
                             dateobs="2025-03-02T00:01:00", seed=2,
                             bad_target_pixel=True)
    refs_mag = {rid: ZP0 - 2.5 * np.log10(rate) for rid, rate in rates.items()
                if rid in ref_ids}
    params = photo_params(ref_ids, rates, refs_mag)
    r = client.post("/api/photometry",
                    files=[("images", ("a.fits", good, "application/fits")),
                           ("images", ("b.fits", bad, "application/fits"))],
                    data={"params": json.dumps(params)})
    assert r.status_code == 200, r.text
    frames = r.json()["frames"]
    assert len(frames) == 2
    oks = {f["frame"]: f for f in frames}
    assert oks[1]["ok"] is True
    assert oks[2]["ok"] is False
    assert oks[2]["target_mag"] is None
    assert "non-finite" in oks[2]["reason"]


def test_bad_exptime_frame_fails(client):
    blob, ref_ids, rates = render_frame(1000.0, 1.0,
                                        dateobs="2025-03-03T00:00:00")
    hdu = fits.open(io.BytesIO(blob))
    hdu[0].header["EXPTIME"] = 0.0
    buf = io.BytesIO()
    hdu.writeto(buf)
    bad = buf.getvalue()
    good, _, _ = render_frame(1000.0, 1.0, dateobs="2025-03-03T00:01:00",
                              seed=5)
    refs_mag = {rid: ZP0 - 2.5 * np.log10(rate) for rid, rate in rates.items()
                if rid in ref_ids}
    params = photo_params(ref_ids, rates, refs_mag)
    r = client.post("/api/photometry",
                    files=[("images", ("a.fits", bad, "application/fits")),
                           ("images", ("b.fits", good, "application/fits"))],
                    data={"params": json.dumps(params)})
    frames = r.json()["frames"]
    bad_rec = next(f for f in frames if f["frame"] == 1)
    assert bad_rec["ok"] is False and "EXPTIME" in bad_rec["reason"]


@pytest.mark.parametrize("override,needle", [
    ({"aperture_radius": 9.0}, "radii"),
    ({"gain": 0.0}, "greater than 0"),
])
def test_param_validation(client, override, needle):
    blob, ref_ids, rates = render_frame(1000.0, 1.0,
                                        dateobs="2025-03-04T00:00:00")
    refs_mag = {rid: 13.0 for rid in ref_ids}
    params = photo_params(ref_ids, rates, refs_mag, **override)
    r = client.post("/api/photometry",
                    files=[("images", ("a.fits", blob, "application/fits")),
                           ("images", ("b.fits", blob, "application/fits"))],
                    data={"params": json.dumps(params)})
    assert r.status_code == 422
    assert needle in r.text


def test_target_must_not_be_reference(client):
    blob, ref_ids, rates = render_frame(1000.0, 1.0,
                                        dateobs="2025-03-04T00:00:00")
    refs_mag = {rid: 13.0 for rid in ref_ids}
    params = photo_params(ref_ids, rates, refs_mag)
    params["references"][0]["id"] = "star000"
    r = client.post("/api/photometry",
                    files=[("images", ("a.fits", blob, "application/fits")),
                           ("images", ("b.fits", blob, "application/fits"))],
                    data={"params": json.dumps(params)})
    assert r.status_code == 422 and "target" in r.text


def test_catalog_order_independence_and_pc_observer_header():
    # Shuffling catalog order must not change the solved geometry.
    blob, _, _ = render_frame(1000.0, 1.0, dateobs="2025-03-05T00:00:00")
    catalog = [{"id": i, "ra": float(r), "dec": float(d)}
               for i, r, d in zip(IDS, RAS, DECS)]
    params = {"catalog": catalog, "center_ra": CENTER_RA,
              "center_dec": CENTER_DEC, "pixel_scale_min": 1.0,
              "pixel_scale_max": 1.6, "rms_max": 0.5}
    c = TestClient(app)
    r1 = c.post("/api/solve",
                files={"image": ("i.fits", blob, "application/fits")},
                data={"params": json.dumps(params)})
    rng = np.random.default_rng(7)
    perm = np.arange(len(catalog))
    rng.shuffle(perm)
    params["catalog"] = [catalog[i] for i in perm]
    r2 = c.post("/api/solve",
                files={"image": ("i.fits", blob, "application/fits")},
                data={"params": json.dumps(params)})
    assert r1.status_code == r2.status_code == 200
    assert abs(r1.json()["rms_arcsec"] - r2.json()["rms_arcsec"]) < 1e-9
    assert r1.json()["n_pairs"] == r2.json()["n_pairs"]
    sid = r1.json()["solve_id"]
    got = c.get(f"/api/solve/{sid}/fits").content
    h = fits.open(io.BytesIO(got))[0].header
    assert h["PC_TEMP"] == pytest.approx(12.3)
