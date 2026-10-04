"""Build a TAN WCS from the solved affine and merge it into the FITS HDU (req. 5)."""
from __future__ import annotations

import re

import numpy as np
from astropy.io import fits

# Old-WCS keywords that must be removed to avoid conflicting solutions.
# Match the WCS grammar precisely (CD1_1, PC2_2, CRPIX1A, ...) instead of
# using bare prefixes: a startswith("PC") test also deleted observer
# keywords such as PC_TEMP or PCTIME, and startswith("CD") hit CDP*.
# Alternate-coordinate corrections (A_ORDER, A_0_2, AP_*, BP_*) match too.
_WCS_KEY_RE = re.compile(
    r"^(?:"
    r"CD[0-9_]{2,4}[A-Z]?"          # CD1_1, CD1_1A
    r"|PC[0-9_]{2,4}[A-Z]?"         # PC1_1, PC2_2A
    r"|CDELT[0-9]+[A-Z]?"           # CDELT1, CDELT1A
    r"|CROTA[0-9]+[A-Z]?"           # CROTA2
    r"|CRPIX[0-9]+[A-Z]?"           # CRPIX1
    r"|CRVAL[0-9]+[A-Z]?"           # CRVAL2A
    r"|CTYPE[0-9]+[A-Z]?"           # CTYPE1
    r"|CUNIT[0-9]+[A-Z]?"           # CNAME1
    r"|CNAME[0-9]+[A-Z]?"           # CUNIT2
    r"|P[VS][0-9]+_[0-9]+[A-Z]?"    # PV1_2, PS2_1
    r"|WCSAXES"
    r"|LONPOLE|LATPOLE|RADESYS|EQUINOX|EPOCH"
    r"|[AB]P?_ORDER|[AB]P?_[0-9]+_[0-9]+"
    r")$"
)


def affine_to_wcs_header(A: np.ndarray, b: np.ndarray,
                         center_ra: float, center_dec: float) -> dict:
    """Convert pixel = A @ tangent_arcsec + b into a TAN WCS header.

    Pixels are 0-based internally; FITS CRPIX is 1-based, hence the +1 shift.
    The inverse affine maps pixel offsets to tangent-plane degrees.
    """
    Ainv = np.linalg.inv(A) / 3600.0  # deg per pixel
    crpix = b + 1.0  # FITS 1-based reference pixel of the tangent point
    return {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": float(center_ra),
        "CRVAL2": float(center_dec),
        "CRPIX1": float(crpix[0]),
        "CRPIX2": float(crpix[1]),
        "CD1_1": float(Ainv[0, 0]),
        "CD1_2": float(Ainv[0, 1]),
        "CD2_1": float(Ainv[1, 0]),
        "CD2_2": float(Ainv[1, 1]),
        "RADESYS": "ICRS",
    }


def strip_old_wcs(header: fits.Header) -> None:
    for key in list(header.keys()):
        if _WCS_KEY_RE.match(key):
            try:
                del header[key]
            except KeyError:
                pass


def solved_fits_bytes(hdu: fits.PrimaryHDU, wcs_header: dict) -> bytes:
    """Return FITS bytes with original pixels/headers plus the new WCS only."""
    import io

    out = fits.PrimaryHDU(data=hdu.data, header=hdu.header.copy())
    strip_old_wcs(out.header)
    for k, v in wcs_header.items():
        out.header[k] = v
    buf = io.BytesIO()
    out.writeto(buf, overwrite=True)
    return buf.getvalue()
