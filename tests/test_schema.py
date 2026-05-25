from schema import LaptopSpecs, UseCaseFitEvaluation


def test_unknown_fields_lists_null_columns() -> None:
    specs = LaptopSpecs(model_name="XPS 13", npu_tops=None, refresh_rate_hz=120)
    assert "npu_tops" in specs.unknown_fields
    assert "model_name" not in specs.unknown_fields
    assert "refresh_rate_hz" not in specs.unknown_fields


def test_unknown_fields_empty_when_complete() -> None:
    specs = LaptopSpecs(
        model_name="X",
        processor_architecture="arm64",
        npu_tops=10.0,
        display_panel_type="OLED",
        tdp_watts=28.0,
        gan_charging_support=True,
        refresh_rate_hz=120,
        weight_kg=1.2,
        screen_size_inches=14.0,
        battery_capacity_wh=57,
        operating_system="Windows 11",
        gpu_type="integrated",
    )
    assert specs.unknown_fields == []


def test_fit_evaluation_model() -> None:
    fit = UseCaseFitEvaluation(
        recommendation="compromise",
        rationale="Good screen; battery may be limiting.",
    )
    assert fit.recommendation == "compromise"
