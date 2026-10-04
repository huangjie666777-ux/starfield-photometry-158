"""Differential aperture photometry on a solved frame.

Pixel coordinates of catalog stars come from the shared plate solution.
Aperture sums include every pixel whose *center* falls inside the circular
aperture.  The local background is estimated robustly from the annulus after
masking pixels contaminated by other stars.  Uncertainties combine Poisson
(source + background), read noise and the uncertainty of the background
estimate itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ApertureResult:
    ok: bool
    reason: str | None = None
    flux_rate: float | None = None     # background-subtracted electrons / s
    flux_rate_err: float | None = None
    background: float | None = None    # ADU / pixel
    background_err: float | None = None
    n_ap: int = 0
    n_bg: int = 0


def _circle_pixels(cx: float, cy: float, radius: float,
                   shape: tuple[int, int]):
    """Return (rows, cols, mask-in-bbox, x0, y0) of pixels whose centers are
    within *radius*, clipped to the frame; plus an in-frame flag."""
    ny, nx = shape
    x0 = int(np.floor(cx - radius))
    x1 = int(np.ceil(cx + radius))
    y0 = int(np.floor(cy - radius))
    y1 = int(np.ceil(cy + radius))
    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)
    xx, yy = np.meshgrid(xs, ys)
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    in_frame = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
    rr, cc = yy[inside & in_frame], xx[inside & in_frame]
    fully_inside = bool(np.all(in_frame[inside]))
    return rr, cc, fully_inside


def _annulus_pixels(cx: float, cy: float, r_in: float, r_out: float,
                    shape: tuple[int, int]):
    ny, nx = shape
    x0 = int(np.floor(cx - r_out))
    x1 = int(np.ceil(cx + r_out))
    y0 = int(np.floor(cy - r_out))
    y1 = int(np.ceil(cy + r_out))
    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)
    xx, yy = np.meshgrid(xs, ys)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (d2 <= r_out ** 2) & (d2 >= r_in ** 2)
    in_frame = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
    sel = ring & in_frame
    return yy[sel], xx[sel]


def _robust_bg(pix: np.ndarray):
    """Median + MAD-sigma background, with a sigma-clip pass for star leaks."""
    med = float(np.median(pix))
    sig = 1.4826 * float(np.median(np.abs(pix - med)))
    if not np.isfinite(sig) or sig <= 0:
        # Degenerate MAD (flat background): keep the median, sigma ~0 means
        # only background-mean uncertainty below matters.
        sig = 0.0
        good = np.isfinite(pix)
    else:
        good = np.isfinite(pix) & (pix < med + 5.0 * sig)
        if good.sum() < max(5, pix.size // 4):
            good = np.isfinite(pix)
        med = float(np.median(pix[good]))
    return med, sig, int(good.sum())


def aperture_photometry(
    data: np.ndarray,
    cx: float,
    cy: float,
    other_positions: np.ndarray,
    aperture_radius: float,
    inner_radius: float,
    outer_radius: float,
    gain: float,
    read_noise: float,
    exptime: float,
    saturation: float | None = None,
) -> ApertureResult:
    """Background-subtracted aperture photometry for one star.

    ``other_positions`` is an (M, 2) array (x, y) of every other catalog
    star; apertures/annuli contaminated by them are flagged ``crowded``.
    Pixel units are ADU; output flux is in electrons per second.
    """
    ny, nx = data.shape
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return ApertureResult(False, "star projects to non-finite pixel")

    # Aperture pixel set (pixel-center criterion).
    rr, cc, fully_inside = _circle_pixels(cx, cy, aperture_radius, data.shape)
    if rr.size == 0:
        return ApertureResult(False, "aperture out of bounds")
    if not fully_inside:
        return ApertureResult(False, "aperture crosses image edge", n_ap=rr.size)

    # Crowding: another catalog star's center falls within the aperture or
    # its own aperture reaches into this aperture.
    if len(other_positions):
        d = np.hypot(other_positions[:, 0] - cx, other_positions[:, 1] - cy)
        if np.any(d < 2.0 * aperture_radius):
            return ApertureResult(False, "crowded", n_ap=rr.size)

    ap_pix = data[rr, cc]
    if not np.all(np.isfinite(ap_pix)):
        return ApertureResult(False, "non-finite pixel in aperture",
                              n_ap=rr.size)
    if saturation is not None and np.any(ap_pix >= saturation):
        return ApertureResult(False, "saturated", n_ap=rr.size)

    # Background annulus, excluding pixels inside any other star's aperture
    # and pixels inside this star's aperture footprint.
    br, bc = _annulus_pixels(cx, cy, inner_radius, outer_radius, data.shape)
    if br.size < 5:
        return ApertureResult(False, "annulus out of bounds", n_ap=rr.size)
    keep = np.ones(br.size, dtype=bool)
    # Drop non-finite annulus pixels explicitly.
    keep &= np.isfinite(data[br, bc])
    if saturation is not None:
        keep &= data[br, bc] < saturation
    if len(other_positions):
        dx = other_positions[:, 0][None, :] - bc[:, None].astype(float)
        dy = other_positions[:, 1][None, :] - br[:, None].astype(float)
        contaminated = np.any(dx ** 2 + dy ** 2
                              < (2.0 * aperture_radius) ** 2, axis=1)
        keep &= ~contaminated
    br, bc = br[keep], bc[keep]
    if br.size < 5:
        return ApertureResult(False, "crowded annulus", n_ap=rr.size)

    bg_pix = data[br, bc].astype(np.float64)
    bg_med, bg_sig, n_clip = _robust_bg(bg_pix)
    n_ap = rr.size
    n_bg = br.size

    # Background-subtracted aperture counts in ADU.
    gross_adu = float(ap_pix.astype(np.float64).sum())
    net_adu = gross_adu - bg_med * n_ap
    if not np.isfinite(net_adu) or net_adu <= 0:
        return ApertureResult(False, "non-positive net flux",
                              background=bg_med, n_ap=n_ap, n_bg=n_bg)

    # Electron-domain variance:
    #   Poisson(source + bg in aperture) + n_ap * RN^2
    # background level uncertainty: n_ap^2 * sigma_bg^2 / n_bg
    # Use clipped MAD sigma when positive; otherwise estimate per-pixel
    # Poisson+RN sigma from the background level itself.
    bg_est_var = bg_sig ** 2 / n_bg if bg_sig > 0 else 0.0
    if bg_sig <= 0:
        # Infer sky+RN scatter in electrons from the median level.
        bg_e = max(bg_med, 0.0) * gain
        bg_sig = np.sqrt(max(bg_e, 0.0) + read_noise ** 2) / gain
        bg_est_var = bg_sig ** 2 / n_bg

    gross_e = gross_adu * gain
    bg_e = max(bg_med, 0.0) * n_ap * gain
    var_e = max(gross_e, 0.0) + bg_e + n_ap * read_noise ** 2 \
        + (n_ap * gain) ** 2 * bg_est_var
    var_e = max(var_e, 0.0)

    flux_rate = net_adu * gain / exptime
    flux_rate_err = np.sqrt(var_e) / exptime
    return ApertureResult(
        ok=True, flux_rate=float(flux_rate),
        flux_rate_err=float(flux_rate_err),
        background=bg_med,
        background_err=float(bg_sig / np.sqrt(n_bg)),
        n_ap=n_ap, n_bg=n_bg)
