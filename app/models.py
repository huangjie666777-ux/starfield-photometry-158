"""Request/response schemas and server-side limits for the plate solver."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Server-side resource limits (requirement 1).
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGE_SIDE = 8192
MAX_CATALOG_STARS = 2000
MAX_DETECTED_SOURCES = 500
MAX_FITS_BYTES = 64 * 1024 * 1024


class CatalogStar(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    ra: float = Field(ge=0.0, lt=360.0, description="ICRS RA in degrees")
    dec: float = Field(ge=-90.0, le=90.0, description="ICRS Dec in degrees")


class SolveParams(BaseModel):
    catalog: List[CatalogStar] = Field(min_length=6, max_length=MAX_CATALOG_STARS)
    center_ra: float = Field(ge=0.0, lt=360.0)
    center_dec: float = Field(ge=-90.0, le=90.0)
    pixel_scale_min: float = Field(gt=0.0, le=3600.0, description="arcsec/pixel")
    pixel_scale_max: float = Field(gt=0.0, le=3600.0, description="arcsec/pixel")
    rms_max: float = Field(gt=0.0, le=3600.0, description="max acceptable RMS, arcsec")
    threshold_sigma: float = Field(default=5.0, gt=0.0, le=100.0)
    saturation: Optional[float] = Field(
        default=None, description="pixels >= this are treated as saturated"
    )
    max_pairs: int = Field(default=200, ge=6, le=1000)

    @field_validator("pixel_scale_max")
    @classmethod
    def _scale_order(cls, v, info):
        lo = info.data.get("pixel_scale_min")
        if lo is not None and v < lo:
            raise ValueError("pixel_scale_max must be >= pixel_scale_min")
        return v

    @field_validator("catalog")
    @classmethod
    def _unique_ids(cls, v):
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog ids must be unique")
        return v


class PairOut(BaseModel):
    id: str
    x: float  # 0-based pixel centroid
    y: float
    residual_arcsec: float


class SolveResponse(BaseModel):
    solve_id: str
    n_pairs: int
    rms_arcsec: float
    mirrored: bool
    pairs: List[PairOut]
    wcs: dict


# ------------------------- Differential photometry -------------------------

MAX_FRAMES = 20


class RefStar(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    mag: float = Field(gt=0.0, lt=40.0, description="known catalog magnitude")
    mag_err: float = Field(default=0.0, ge=0.0, lt=10.0,
                           description="known magnitude uncertainty (0 if n/a)")


class PhotoParams(BaseModel):
    catalog: List[CatalogStar] = Field(min_length=6, max_length=MAX_CATALOG_STARS)
    center_ra: float = Field(ge=0.0, lt=360.0)
    center_dec: float = Field(ge=-90.0, le=90.0)
    pixel_scale_min: float = Field(gt=0.0, le=3600.0, description="arcsec/pixel")
    pixel_scale_max: float = Field(gt=0.0, le=3600.0, description="arcsec/pixel")
    rms_max: float = Field(gt=0.0, le=3600.0, description="max plate RMS, arcsec")
    threshold_sigma: float = Field(default=5.0, gt=0.0, le=100.0)
    saturation: Optional[float] = Field(default=None)
    max_pairs: int = Field(default=200, ge=6, le=1000)

    target_id: str = Field(min_length=1, max_length=64)
    references: List[RefStar] = Field(min_length=3, max_length=100)
    aperture_radius: float = Field(gt=0.0, le=1000.0, description="pixels")
    inner_radius: float = Field(gt=0.0, le=2000.0, description="annulus inner, px")
    outer_radius: float = Field(gt=0.0, le=2000.0, description="annulus outer, px")
    gain: float = Field(gt=0.0, le=1e6, description="electrons per ADU")
    read_noise: float = Field(ge=0.0, le=1e6, description="electrons per pixel")

    @field_validator("catalog")
    @classmethod
    def _unique_ids(cls, v):
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog ids must be unique")
        return v

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.pixel_scale_max < self.pixel_scale_min:
            raise ValueError("pixel_scale_max must be >= pixel_scale_min")
        if not (self.aperture_radius < self.inner_radius <= self.outer_radius):
            raise ValueError(
                "radii must satisfy aperture_radius < inner_radius <= outer_radius")
        cat_ids = {s.id for s in self.catalog}
        if self.target_id not in cat_ids:
            raise ValueError(f"target_id {self.target_id!r} not found in catalog")
        ref_ids = [r.id for r in self.references]
        if len(set(ref_ids)) != len(ref_ids):
            raise ValueError("reference star ids must be unique")
        bad = [i for i in ref_ids if i not in cat_ids]
        if bad:
            raise ValueError(f"reference stars missing from catalog: {bad}")
        if self.target_id in ref_ids:
            raise ValueError("target star must not also be a reference star")
        return self


class RefOut(BaseModel):
    id: str
    used: bool
    reason: Optional[str] = None


class FrameResult(BaseModel):
    frame: int
    ok: bool
    reason: Optional[str] = None
    mjd_mid: Optional[float] = None
    flux_rate: Optional[float] = None
    flux_rate_err: Optional[float] = None
    target_mag: Optional[float] = None
    target_mag_err: Optional[float] = None
    zero_point: Optional[float] = None
    zero_point_err: Optional[float] = None
    n_refs_used: int = 0
    references: List[RefOut] = []


class PhotoResponse(BaseModel):
    target_id: str
    n_frames: int
    n_ok: int
    frames: List[FrameResult]
