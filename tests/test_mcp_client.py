from mcp_client import (
    laptop_specs_to_insert_sql,
    parse_read_rows,
    row_to_laptop_specs,
    sql_literal,
)
from mcp.types import CallToolResult, TextContent
from schema import LaptopSpecs


def test_sql_literal_escapes_quotes() -> None:
    assert sql_literal("it's") == "'it''s'"


def test_parse_read_rows_from_tool_text() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="[{'slug_id': 'x', 'model_name': 'Y'}]")],
        isError=False,
    )
    rows = parse_read_rows(result)
    assert rows == [{"slug_id": "x", "model_name": "Y"}]


def test_row_to_laptop_specs_maps_columns() -> None:
    specs = row_to_laptop_specs(
        {
            "slug_id": "dell-xps",
            "model_name": "XPS",
            "processor_architecture": "x86",
            "npu_tops": 10.0,
            "display_panel_type": "OLED",
            "thermal_design_power_watts": 28,
            "usb_c_gan_charging_support": 1,
            "refresh_rate_hz": 120,
            "weight_kg": 1.27,
            "screen_size_inches": 13.4,
            "battery_capacity_wh": 52,
            "operating_system": "macOS",
            "gpu_type": "integrated",
            "unknown_fields_json": "[]",
        },
    )
    assert specs.slug_id == "dell-xps"
    assert specs.tdp_watts == 28.0
    assert specs.gan_charging_support is True
    assert specs.refresh_rate_hz == 120
    assert specs.weight_kg == 1.27
    assert specs.gpu_type == "integrated"


def test_laptop_specs_to_insert_sql_includes_slug() -> None:
    specs = LaptopSpecs(
        slug_id="abc-1",
        model_name="Test",
        tdp_watts=15.0,
        gan_charging_support=False,
    )
    sql = laptop_specs_to_insert_sql(specs)
    assert "INSERT OR REPLACE INTO laptops" in sql
    assert "'abc-1'" in sql
    assert "thermal_design_power_watts" in sql
    assert "refresh_rate_hz" in sql


def test_laptop_specs_to_insert_sql_serializes_refresh_rate() -> None:
    specs = LaptopSpecs(slug_id="x", refresh_rate_hz=165)
    sql = laptop_specs_to_insert_sql(specs)
    assert "165" in sql
