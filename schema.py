"""Pydantic models for extracted laptop specs and use-case fit evaluation."""

from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

FitRecommendation = Literal[
    "recommended",
    "not_recommended",
    "compromise",
    "insufficient_data",
]
"""Allowed fit outcomes produced by the evaluate_use_case node."""


class LaptopSpecs(BaseModel):
    """Structured laptop hardware specifications extracted from retail pages.

    All fields are optional because retail copy may omit values; null fields
    trigger the web-search retry loop via ``unknown_fields``.
    """

    model_name: Optional[str] = Field(
        default=None,
        description="Product or laptop model name as listed on the page.",
    )
    processor_architecture: Optional[str] = Field(
        default=None,
        description="CPU architecture (e.g. x86-64, ARM64, hybrid P-cores/E-cores).",
    )
    npu_tops: Optional[float] = Field(
        default=None,
        description="Neural Processing Unit peak performance in TOPS.",
    )
    display_panel_type: Optional[str] = Field(
        default=None,
        description="Display panel technology (e.g. IPS, OLED, Mini-LED).",
    )
    tdp_watts: Optional[float] = Field(
        default=None,
        description="Processor or platform Thermal Design Power in watts.",
    )
    gan_charging_support: Optional[bool] = Field(
        default=None,
        description="Whether GaN fast charging is included or explicitly supported.",
    )
    refresh_rate_hz: Optional[int] = Field(
        default=None,
        description="Display refresh rate in Hz.",
    )
    weight_kg: Optional[float] = Field(
        default=None,
        description="Laptop weight in kilograms.",
    )
    screen_size_inches: Optional[float] = Field(
        default=None,
        description="Diagonal screen size in inches.",
    )
    battery_capacity_wh: Optional[int] = Field(
        default=None,
        description="Battery capacity in watt-hours (Wh).",
    )
    operating_system: Optional[str] = Field(
        default=None,
        description="Preinstalled operating system (e.g. Windows 11 Home, macOS).",
    )
    gpu_type: Optional[Literal["dedicated", "integrated", "hybrid"]] = Field(
        default=None,
        description=(
            "GPU class: dedicated discrete GPU, integrated only, or hybrid (switchable)."
        ),
    )
    slug_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable cross-retailer identity key (lowercase slug derived from model or MPN)."
        ),
    )
    part_number_mpn: Optional[str] = Field(
        default=None,
        description="Manufacturer part number or SKU when stated on the page or URL.",
    )

    _CACHE_IDENTITY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"slug_id", "part_number_mpn"},
    )

    @property
    def unknown_fields(self) -> list[str]:
        """Return schema field names that are still null.

        Returns:
            List of attribute names (e.g. ``npu_tops``) with value ``None``.
            Empty when every field is populated. Identity fields are excluded.
        """
        return [
            name
            for name, value in self.model_dump().items()
            if value is None and name not in self._CACHE_IDENTITY_FIELDS
        ]


class UseCaseFitEvaluation(BaseModel):
    """Gemini judgment of how well laptop specs match the user's use case.

    Attributes:
        recommendation: One of ``recommended``, ``not_recommended``,
            ``compromise``, or ``insufficient_data``.
        rationale: Short explanation referencing specs and use case.
    """

    recommendation: FitRecommendation = Field(
        description=(
            "recommended: strong fit; not_recommended: poor fit; "
            "compromise: acceptable with tradeoffs; insufficient_data: cannot judge"
        ),
    )
    rationale: str = Field(
        description="Brief explanation (2–3 sentences) citing specs vs use case.",
    )
