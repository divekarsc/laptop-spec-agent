from typing import Literal, Optional

from pydantic import BaseModel, Field

FitRecommendation = Literal[
    "recommended",
    "not_recommended",
    "compromise",
    "insufficient_data",
]


class LaptopSpecs(BaseModel):
    """Structured laptop hardware specifications extracted from retail pages."""

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

    @property
    def unknown_fields(self) -> list[str]:
        """Field names that are still null after extraction."""
        return [name for name, value in self.model_dump().items() if value is None]


class UseCaseFitEvaluation(BaseModel):
    """Whether the laptop meets the user's stated use case."""

    recommendation: FitRecommendation = Field(
        description=(
            "recommended: strong fit; not_recommended: poor fit; "
            "compromise: acceptable with tradeoffs; insufficient_data: cannot judge"
        ),
    )
    rationale: str = Field(
        description="Brief explanation (2–3 sentences) citing specs vs use case.",
    )
