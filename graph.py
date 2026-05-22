import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from typing import Any, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from errors import (
    MAX_USE_CASE_WORDS,
    AgentError,
    ValidationError,
    cli_exit_message,
    empty_page_content_error,
    error_message_for_state,
    summarize_run_failure,
    validate_product_url,
    validate_use_case,
)
from llm_config import get_fit_evaluation_llm, get_gemini_model_name, get_structured_llm
from logging_config import get_system_logger, get_user_logger, setup_logging
from schema import LaptopSpecs, UseCaseFitEvaluation
from state import AgentState
from tools import fetch_page_text, search_web

load_dotenv()
setup_logging()

user_log = get_user_logger()
system_log = get_system_logger()

_EXIT_VALIDATION = 2
_EXIT_RUNTIME = 1

_MAX_RETRIES = 2
_CONTENT_LIMIT = 50_000

_FIELD_SEARCH_LABELS: dict[str, str] = {
    "model_name": "model name",
    "processor_architecture": "processor architecture",
    "npu_tops": "NPU TOPS",
    "display_panel_type": "display panel type",
    "tdp_watts": "TDP watts",
    "gan_charging_support": "GaN charging",
    "refresh_rate_hz": "refresh rate Hz",
}

_EXTRACT_PROMPT = """Extract laptop hardware specifications from the retail product page text below.
Use only information present in the text; leave fields null when not stated or unclear.

Page text:
{content}"""

_STREAM_MODES = ("updates", "custom", "values")

_NODE_STATUS: dict[str, str] = {
    "extract_page": "Scraping product page…",
    "parse_specs": "Parsing specs with Gemini…",
    "search_missing_specs": "Searching the web for missing specs…",
    "evaluate_use_case": "Evaluating fit for your use case…",
}

_FIT_EVAL_PROMPT = """You are advising a laptop buyer. Given structured hardware specs and the user's use case, decide how well this laptop fits.

Use case (user's words):
{use_case}

Laptop specs (JSON):
{specs}

Choose one recommendation:
- recommended: strong fit for the use case
- not_recommended: poor fit; important requirements likely unmet
- compromise: workable with clear tradeoffs (battery, performance, screen, etc.)
- insufficient_data: too many unknown specs to judge fairly

Base your answer only on the specs and use case provided. Be concise in the rationale."""

_RECOMMENDATION_LABELS = {
    "recommended": "Recommended",
    "not_recommended": "Not recommended",
    "compromise": "Compromise",
    "insufficient_data": "Insufficient data",
}


class AgentStatus(TypedDict):
    """User-facing status event emitted while the graph runs."""

    message: str
    node: NotRequired[str]
    phase: NotRequired[str]
    final_state: NotRequired[AgentState]


def _emit_status(message: str, *, node: str | None = None, phase: str | None = None) -> None:
    payload: dict[str, Any] = {"message": message}
    if node:
        payload["node"] = node
    if phase:
        payload["phase"] = phase
    try:
        get_stream_writer()(payload)
    except RuntimeError:
        pass


def _initial_state(url: str, use_case: str) -> AgentState:
    return {"url": url, "use_case": use_case, "error": "", "retry_count": 0}


def prompt_use_case_interactive() -> str:
    """Read a use case from stdin (max 25 words)."""
    print(
        f"\nDescribe your laptop use case in {MAX_USE_CASE_WORDS} words or fewer, "
        "then press Enter:"
    )
    return validate_use_case(input().strip())


def resolve_use_case(cli_value: str | None) -> str:
    """Use --use-case flag or prompt interactively."""
    if cli_value is not None:
        return validate_use_case(cli_value)
    return prompt_use_case_interactive()


def _format_search_query(model_name: str | None, field_name: str) -> str:
    label = _FIELD_SEARCH_LABELS.get(field_name, field_name.replace("_", " "))
    product = (model_name or "laptop").strip()
    return f"{product} {label} specs"


async def extract_page_node(state: AgentState) -> dict:
    if state.get("error"):
        system_log.debug("extract_page_node skipped: prior error=%s", state["error"])
        return {}

    url = state["url"]
    _emit_status(_NODE_STATUS["extract_page"], node="extract_page", phase="scrape")
    user_log.info("Step 1: Scraping product page.")
    system_log.info("extract_page_node start url=%s", url)

    try:
        raw_content = await fetch_page_text.ainvoke({"url": url})
        _emit_status(
            f"Page loaded ({len(raw_content):,} characters).",
            node="extract_page",
            phase="scrape",
        )
        system_log.info(
            "extract_page_node success url=%s raw_content_chars=%d",
            url,
            len(raw_content),
        )
        return {"raw_content": raw_content, "error": ""}
    except Exception as exc:
        message = error_message_for_state(exc, phase="extract_page", url=url)
        _emit_status(message, node="extract_page", phase="error")
        user_log.error("Page scrape failed: %s", message)
        system_log.error("extract_page_node failed url=%s", url, exc_info=exc)
        return {"error": message}


async def parse_specs_node(state: AgentState) -> dict:
    if state.get("error"):
        system_log.debug("parse_specs_node skipped: error=%s", state["error"])
        return {}

    raw_content = state.get("raw_content", "").strip()
    if not raw_content:
        message = empty_page_content_error().formatted()
        _emit_status(message, node="parse_specs", phase="error")
        user_log.error(message)
        system_log.error("parse_specs_node aborted: empty raw_content")
        return {"error": message}

    retry_count = state.get("retry_count", 0)
    _emit_status(
        f"Parsing specs with Gemini (attempt {retry_count + 1})…",
        node="parse_specs",
        phase="extract",
    )
    user_log.info(
        "Parsing specs with Gemini (attempt %d)…",
        retry_count + 1,
    )
    system_log.info(
        "parse_specs_node start model=%s retry_count=%d prompt_chars=%d",
        get_gemini_model_name(),
        retry_count,
        min(len(raw_content), _CONTENT_LIMIT),
    )

    try:
        specs = await get_structured_llm().ainvoke(
            [
                HumanMessage(
                    content=_EXTRACT_PROMPT.format(
                        content=raw_content[:_CONTENT_LIMIT],
                    ),
                ),
            ],
        )
        unknown = specs.unknown_fields
        if unknown:
            _emit_status(
                f"Missing fields: {', '.join(unknown)}",
                node="parse_specs",
                phase="validate",
            )
            user_log.info("Missing fields after parse: %s", ", ".join(unknown))
        else:
            _emit_status("All spec fields populated.", node="parse_specs", phase="validate")
            user_log.info("All spec fields populated.")
        system_log.info(
            "parse_specs_node success specs=%s unknown=%s",
            specs.model_dump_json(),
            unknown,
        )
        return {"specs": specs, "error": ""}
    except Exception as exc:
        message = error_message_for_state(exc, phase="parse_specs")
        _emit_status(message, node="parse_specs", phase="error")
        user_log.error("Spec parsing failed: %s", message)
        system_log.exception("parse_specs_node failed")
        return {"error": message}


async def search_missing_specs_node(state: AgentState) -> dict:
    if state.get("error"):
        system_log.debug("search_missing_specs_node skipped: error=%s", state["error"])
        return {}

    specs = state.get("specs")
    if not specs or not specs.unknown_fields:
        system_log.debug("search_missing_specs_node skipped: nothing to search")
        return {}

    model_name = specs.model_name
    unknown = specs.unknown_fields
    _emit_status(
        f"Searching for {len(unknown)} missing field(s)…",
        node="search_missing_specs",
        phase="search",
    )
    user_log.info(
        "Searching the web for %d missing field(s): %s",
        len(unknown),
        ", ".join(unknown),
    )
    system_log.info(
        "search_missing_specs_node start model_name=%s fields=%s",
        model_name,
        unknown,
    )

    raw_content = state.get("raw_content", "")
    search_sections: list[str] = []

    try:
        for field_name in unknown:
            query = _format_search_query(model_name, field_name)
            _emit_status(f"Web search: {query}", node="search_missing_specs", phase="search")
            user_log.info("Query: %s", query)
            try:
                results = await search_web.ainvoke({"query": query})
                search_sections.append(
                    f"\n\n--- Web search: {query} ---\n{results}"
                )
            except Exception as exc:
                field_error = error_message_for_state(
                    exc,
                    phase="search_missing_specs",
                    query=query,
                )
                system_log.warning(
                    "search failed field=%s query=%s error=%s",
                    field_name,
                    query,
                    field_error,
                )
                search_sections.append(
                    f"\n\n--- Web search failed: {query} ---\n{field_error}"
                )

        enriched = raw_content + "".join(search_sections)
        retry_count = state.get("retry_count", 0) + 1
        _emit_status(
            f"Search round {retry_count}/{_MAX_RETRIES} complete; re-parsing specs…",
            node="search_missing_specs",
            phase="search",
        )
        user_log.info(
            "Enriched content with search results (retry %d/%d).",
            retry_count,
            _MAX_RETRIES,
        )
        system_log.info(
            "search_missing_specs_node success retry_count=%d added_chars=%d",
            retry_count,
            len(enriched) - len(raw_content),
        )
        return {
            "raw_content": enriched,
            "retry_count": retry_count,
            "error": "",
        }
    except Exception as exc:
        message = error_message_for_state(exc, phase="search_missing_specs")
        _emit_status(message, node="search_missing_specs", phase="error")
        user_log.error("Web search step failed: %s", message)
        system_log.exception("search_missing_specs_node failed")
        return {"error": message}


def route_after_parse(state: AgentState) -> str:
    if state.get("error"):
        system_log.info("route_after_parse -> evaluate_use_case (prior error)")
        return "evaluate_use_case"

    specs = state.get("specs")
    if not specs:
        system_log.info("route_after_parse -> evaluate_use_case (no specs)")
        return "evaluate_use_case"

    retry_count = state.get("retry_count", 0)
    if specs.unknown_fields and retry_count < _MAX_RETRIES:
        system_log.info(
            "route_after_parse -> search_missing_specs unknown=%s retry_count=%d",
            specs.unknown_fields,
            retry_count,
        )
        return "search_missing_specs"

    system_log.info(
        "route_after_parse -> evaluate_use_case unknown=%s retry_count=%d",
        specs.unknown_fields,
        retry_count,
    )
    return "evaluate_use_case"


async def evaluate_use_case_node(state: AgentState) -> dict:
    if state.get("error"):
        system_log.debug(
            "evaluate_use_case_node skipped: prior error=%s",
            state["error"],
        )
        return {}

    use_case = state.get("use_case", "").strip()
    if not use_case:
        message = "Use case is missing from agent state."
        user_log.error(message)
        return {"error": message}

    specs = state.get("specs")
    if not specs:
        message = "Cannot evaluate use case without laptop specs."
        user_log.error(message)
        return {"error": message}

    _emit_status(
        _NODE_STATUS["evaluate_use_case"],
        node="evaluate_use_case",
        phase="fit",
    )
    user_log.info("Evaluating fit for use case: %s", use_case)
    system_log.info(
        "evaluate_use_case_node start use_case=%s specs=%s",
        use_case,
        specs.model_dump_json(),
    )

    try:
        evaluation = await get_fit_evaluation_llm().ainvoke(
            [
                HumanMessage(
                    content=_FIT_EVAL_PROMPT.format(
                        use_case=use_case,
                        specs=specs.model_dump_json(indent=2),
                    ),
                ),
            ],
        )
        label = _RECOMMENDATION_LABELS.get(
            evaluation.recommendation,
            evaluation.recommendation,
        )
        _emit_status(
            f"Fit evaluation: {label}",
            node="evaluate_use_case",
            phase="fit",
        )
        user_log.info("%s — %s", label, evaluation.rationale)
        system_log.info(
            "evaluate_use_case_node success recommendation=%s",
            evaluation.recommendation,
        )
        return {"fit_evaluation": evaluation, "error": ""}
    except Exception as exc:
        message = error_message_for_state(exc, phase="evaluate_use_case")
        _emit_status(message, node="evaluate_use_case", phase="error")
        user_log.error("Use case evaluation failed: %s", message)
        system_log.exception("evaluate_use_case_node failed")
        return {"error": message}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_page", extract_page_node)
    workflow.add_node("parse_specs", parse_specs_node)
    workflow.add_node("search_missing_specs", search_missing_specs_node)
    workflow.add_node("evaluate_use_case", evaluate_use_case_node)
    workflow.add_edge(START, "extract_page")
    workflow.add_edge("extract_page", "parse_specs")
    workflow.add_conditional_edges(
        "parse_specs",
        route_after_parse,
        {
            "search_missing_specs": "search_missing_specs",
            "evaluate_use_case": "evaluate_use_case",
        },
    )
    workflow.add_edge("search_missing_specs", "parse_specs")
    workflow.add_edge("evaluate_use_case", END)
    return workflow.compile()


def _status_from_custom(payload: Any) -> AgentStatus | None:
    if isinstance(payload, dict) and payload.get("message"):
        status: AgentStatus = {"message": str(payload["message"])}
        if node := payload.get("node"):
            status["node"] = str(node)
        if phase := payload.get("phase"):
            status["phase"] = str(phase)
        return status
    if isinstance(payload, str):
        return {"message": payload}
    return None


def _status_from_node_update(node: str, update: dict[str, Any] | None) -> AgentStatus | None:
    if not update:
        return None

    if update.get("error"):
        return {"message": f"Error: {update['error']}", "node": node, "phase": "error"}

    if node == "parse_specs" and "specs" in update:
        specs = update["specs"]
        if isinstance(specs, LaptopSpecs):
            unknown = specs.unknown_fields
            if unknown:
                return {
                    "message": f"Parsed specs; still missing: {', '.join(unknown)}",
                    "node": node,
                    "phase": "validate",
                }
            return {"message": "Parsed specs; all fields filled.", "node": node, "phase": "validate"}

    if node == "evaluate_use_case" and update and "fit_evaluation" in update:
        fit = update["fit_evaluation"]
        if isinstance(fit, UseCaseFitEvaluation):
            label = _RECOMMENDATION_LABELS.get(fit.recommendation, fit.recommendation)
            return {
                "message": f"Fit: {label} — {fit.rationale}",
                "node": node,
                "phase": "fit",
            }

    return None


def _log_status(status: AgentStatus) -> None:
    user_log.info("%s", status["message"])
    system_log.debug(
        "stream status node=%s phase=%s message=%s",
        status.get("node"),
        status.get("phase"),
        status["message"],
    )


async def stream_agent(url: str, use_case: str) -> AsyncIterator[AgentStatus]:
    """Yield user-facing status events while the graph runs via ``astream``."""
    graph = build_graph()
    final_state: AgentState | None = None

    _log_status(
        {"message": f"Starting agent for {url}", "phase": "start"},
    )
    _log_status({"message": f"Use case: {use_case}", "phase": "start"})
    system_log.info("stream_agent start url=%s use_case=%s", url, use_case)

    try:
        async for mode, chunk in graph.astream(
            _initial_state(url, use_case),
            stream_mode=list(_STREAM_MODES),
        ):
            if mode == "values":
                final_state = chunk
                continue

            if mode == "custom":
                if status := _status_from_custom(chunk):
                    _log_status(status)
                    yield status
                continue

            if mode == "updates":
                for node, update in chunk.items():
                    if node.startswith("__"):
                        continue
                    if status := _status_from_node_update(node, update):
                        _log_status(status)
                        yield status
    except Exception as exc:
        message = error_message_for_state(exc, phase="graph")
        system_log.exception("stream_agent graph execution failed url=%s", url)
        final_state = {
            **_initial_state(url, use_case),
            "error": message,
        }
        failed: AgentStatus = {
            "message": f"Agent failed: {message}",
            "phase": "done",
            "final_state": final_state,
        }
        _log_status(failed)
        yield failed
        return

    if final_state is None:
        final_state = _initial_state(url, use_case)

    if error := final_state.get("error"):
        done: AgentStatus = {
            "message": f"Agent failed: {error}",
            "phase": "done",
            "final_state": final_state,
        }
    elif fit := final_state.get("fit_evaluation"):
        if isinstance(fit, UseCaseFitEvaluation):
            label = _RECOMMENDATION_LABELS.get(fit.recommendation, fit.recommendation)
            done = {
                "message": f"Done. Fit evaluation: {label}",
                "phase": "done",
                "final_state": final_state,
            }
        else:
            done = {
                "message": "Agent finished.",
                "phase": "done",
                "final_state": final_state,
            }
    elif final_state.get("specs"):
        specs = final_state["specs"]
        if isinstance(specs, LaptopSpecs) and specs.unknown_fields:
            done = {
                "message": (
                    "Specs retrieved (partial); fit evaluation was not completed. "
                    f"Missing: {', '.join(specs.unknown_fields)}"
                ),
                "phase": "done",
                "final_state": final_state,
            }
        else:
            done = {
                "message": "Specs retrieved; fit evaluation was not completed.",
                "phase": "done",
                "final_state": final_state,
            }
    else:
        done = {
            "message": summarize_run_failure(final_state),
            "phase": "done",
            "final_state": final_state,
        }

    _log_status(done)
    yield done
    system_log.info("stream_agent finished url=%s", url)


async def run_agent(url: str, use_case: str) -> AgentState:
    """Run the agent, streaming status via ``astream``, and return the final state."""
    result: AgentState | None = None
    async for status in stream_agent(url, use_case):
        if status.get("phase") == "done":
            result = status.get("final_state")
    return result if result is not None else _initial_state(url, use_case)


def _print_fit_evaluation(evaluation: UseCaseFitEvaluation) -> None:
    label = _RECOMMENDATION_LABELS.get(
        evaluation.recommendation,
        evaluation.recommendation,
    )
    print("\n--- Use case fit ---")
    print(f"Recommendation: {label}")
    print(f"Rationale: {evaluation.rationale}")
    print(evaluation.model_dump_json(indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract laptop hardware specs from a retail product page.",
    )
    parser.add_argument(
        "url",
        help="Product page URL to scrape and parse",
    )
    parser.add_argument(
        "--use-case",
        metavar="TEXT",
        help=(
            f"How you will use the laptop (max {MAX_USE_CASE_WORDS} words). "
            "If omitted, you will be prompted after the URL is validated."
        ),
    )
    return parser.parse_args()


def _exit_cli(message: str, *, code: int) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    try:
        args = _parse_args()
        product_url = validate_product_url(args.url)
    except ValidationError as exc:
        _exit_cli(exc.formatted(), code=_EXIT_VALIDATION)

    async def _main() -> None:
        user_log.info("System logs: logs/system.log")
        use_case = resolve_use_case(args.use_case)
        user_log.info("Use case (%d words): %s", len(use_case.split()), use_case)

        try:
            get_structured_llm()
            get_fit_evaluation_llm()
        except AgentError as exc:
            raise exc

        result = await run_agent(product_url, use_case)

        if result.get("error"):
            raise AgentError(
                str(result["error"]),
                code="run_failed",
            )

        specs = result.get("specs")
        if not specs:
            raise AgentError(
                summarize_run_failure(result),
                code="no_specs",
            )

        print("\n--- Laptop specs ---")
        print(specs.model_dump_json(indent=2))
        if specs.unknown_fields:
            user_log.warning(
                "Partial specs — still missing after retries: %s",
                ", ".join(specs.unknown_fields),
            )

        fit = result.get("fit_evaluation")
        if fit:
            _print_fit_evaluation(fit)
        else:
            user_log.warning("Fit evaluation was not produced.")

    try:
        asyncio.run(_main())
    except ValidationError as exc:
        _exit_cli(exc.formatted(), code=_EXIT_VALIDATION)
    except AgentError as exc:
        _exit_cli(exc.formatted(), code=_EXIT_RUNTIME)
    except KeyboardInterrupt:
        _exit_cli("Interrupted.", code=130)
    except Exception as exc:
        system_log.exception("CLI unhandled exception")
        _exit_cli(cli_exit_message(exc), code=_EXIT_RUNTIME)
