from schema import LaptopSpecs, UseCaseFitEvaluation
from graph import (
    _format_search_query,
    _initial_state,
    _status_from_custom,
    _status_from_node_update,
    build_graph,
    route_after_check_cache,
    route_after_parse,
)


def _state(**kwargs) -> dict:
    base = _initial_state("https://example.com/laptop", "portable coding")
    base.update(kwargs)
    return base


class TestFormatSearchQuery:
    def test_uses_model_name_and_field_label(self) -> None:
        q = _format_search_query("Dell XPS 13", "npu_tops")
        assert q == "Dell XPS 13 NPU TOPS specs"

    def test_fallback_product_name(self) -> None:
        q = _format_search_query(None, "refresh_rate_hz")
        assert q == "laptop refresh rate Hz specs"


class TestRouteAfterCheckCache:
    def test_cache_hit_skips_scrape_and_goes_to_fit(self) -> None:
        state = _state(
            specs=LaptopSpecs(model_name="X", slug_id="x"),
            retry_count=0,
        )
        assert route_after_check_cache(state) == "evaluate_use_case"

    def test_cache_miss_routes_to_extract(self) -> None:
        assert route_after_check_cache(_state()) == "extract_page"


class TestRouteAfterParse:
    def test_routes_to_search_when_missing_and_retries_remain(self) -> None:
        state = _state(
            specs=LaptopSpecs(model_name="X"),
            retry_count=0,
        )
        assert route_after_parse(state) == "search_missing_specs"

    def test_routes_to_fit_when_retries_exhausted(self) -> None:
        state = _state(
            specs=LaptopSpecs(model_name="X"),
            retry_count=2,
        )
        assert route_after_parse(state) == "evaluate_use_case"

    def test_routes_to_fit_when_specs_complete(self) -> None:
        state = _state(
            specs=LaptopSpecs(
                model_name="X",
                processor_architecture="x86",
                npu_tops=1.0,
                display_panel_type="IPS",
                tdp_watts=15.0,
                gan_charging_support=False,
                refresh_rate_hz=60,
                weight_kg=1.5,
                screen_size_inches=15.6,
                battery_capacity_wh=60,
                operating_system="Windows 11",
                gpu_type="dedicated",
            ),
            retry_count=0,
        )
        assert route_after_parse(state) == "evaluate_use_case"

    def test_routes_to_fit_on_prior_error(self) -> None:
        state = _state(error="scrape failed")
        assert route_after_parse(state) == "evaluate_use_case"

    def test_routes_to_fit_without_specs(self) -> None:
        assert route_after_parse(_state()) == "evaluate_use_case"


class TestStatusHelpers:
    def test_custom_payload(self) -> None:
        status = _status_from_custom(
            {"message": "Parsing…", "node": "parse_specs", "phase": "extract"},
        )
        assert status is not None
        assert status["message"] == "Parsing…"
        assert status["node"] == "parse_specs"

    def test_node_update_error(self) -> None:
        status = _status_from_node_update("parse_specs", {"error": "LLM down"})
        assert status is not None
        assert "LLM down" in status["message"]
        assert status["phase"] == "error"

    def test_node_update_fit_evaluation(self) -> None:
        fit = UseCaseFitEvaluation(
            recommendation="recommended",
            rationale="Meets portability needs.",
        )
        status = _status_from_node_update(
            "evaluate_use_case",
            {"fit_evaluation": fit},
        )
        assert status is not None
        assert "Recommended" in status["message"]


class TestBuildGraph:
    def test_includes_expected_nodes(self) -> None:
        graph = build_graph()
        node_names = {n for n in graph.get_graph().nodes if not n.startswith("__")}
        assert node_names == {
            "check_cache",
            "extract_page",
            "parse_specs",
            "save_to_cache",
            "search_missing_specs",
            "evaluate_use_case",
        }
