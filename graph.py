"""LangGraph workflow, CLI, and streaming status for the laptop spec agent."""

import argparse
import asyncio
import os
import re
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
from prompts import build_extract_specs_prompt, build_fit_evaluation_prompt
from mcp_client import (
    fetch_laptop_by_slug,
    lookup_slug_by_mpn,
    lookup_slug_by_url,
    mcp_sqlite_session,
    save_laptop_cache,
)
from mpn import extract_mpn_from_url
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
    "weight_kg": "weight kg",
    "screen_size_inches": "screen size inches",
    "battery_capacity_wh": "battery Wh",
    "operating_system": "operating system",
    "gpu_type": "GPU dedicated integrated",
}

_STREAM_MODES = ("updates", "custom", "values")

_NODE_STATUS: dict[str, str] = {
    "check_cache": "Checking spec cache…",
    "extract_page": "Scraping product page…",
    "parse_specs": "Parsing specs with Gemini…",
    "save_to_cache": "Saving specs to cache…",
    "search_missing_specs": "Searching the web for missing specs…",
    "evaluate_use_case": "Evaluating fit for your use case…",
}

_RECOMMENDATION_LABELS = {
    "recommended": "Recommended",
    "not_recommended": "Not recommended",
    "compromise": "Compromise",
    "insufficient_data": "Insufficient data",
}


class AgentStatus(TypedDict):
    """User-facing status event emitted while the graph runs via ``astream``.

    Attributes:
        message: Human-readable progress or outcome text.
        node: LangGraph node name when applicable.
        phase: Coarse step (e.g. ``scrape``, ``extract``, ``fit``, ``done``).
        final_state: Full ``AgentState`` on the terminal ``done`` event only.
    """

    message: str
    node: NotRequired[str]
    phase: NotRequired[str]
    final_state: NotRequired[AgentState]


def _emit_status(message: str, *, node: str | None = None, phase: str | None = None) -> None:
    """Push a progress message to LangGraph custom stream consumers.

    Args:
        message: Text shown on the console during streaming.
        node: Optional graph node name for context.
        phase: Optional phase label (``scrape``, ``search``, ``fit``, etc.).

    Returns:
        None. Silently no-ops when not executing inside a streaming graph.
    """
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
    """Build the state dict passed to ``graph.ainvoke`` / ``graph.astream``.

    Args:
        url: Validated product page URL.
        use_case: Validated use-case string (≤25 words).

    Returns:
        ``AgentState`` with ``retry_count`` 0 and empty ``error``.
    """
    return {"url": url, "use_case": use_case, "error": "", "retry_count": 0}


def prompt_use_case_interactive() -> str:
    """Read and validate a use case from stdin.

    Prints a prompt, reads one line, and enforces the 25-word limit.

    Returns:
        Normalized use-case string.

    Raises:
        ValidationError: If input is empty or too long.
    """
    print(
        f"\nDescribe your laptop use case in {MAX_USE_CASE_WORDS} words or fewer, "
        "then press Enter:"
    )
    return validate_use_case(input().strip())


def resolve_use_case(cli_value: str | None) -> str:
    """Obtain use case from CLI flag or interactive prompt.

    Args:
        cli_value: Value of ``--use-case`` if provided; ``None`` triggers prompt.

    Returns:
        Validated use-case string.
    """
    if cli_value is not None:
        return validate_use_case(cli_value)
    return prompt_use_case_interactive()


def _slugify_identity(value: str) -> str:
    """Convert a model name or MPN into a stable lowercase slug id.

    Args:
        value: Raw identity string (model name, MPN, etc.).

    Returns:
        Slug with letters, digits, and hyphens (max 80 chars), or ``laptop-unknown``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] if slug else "laptop-unknown"


def _ensure_identity_fields(specs: LaptopSpecs, url: str) -> LaptopSpecs:
    """Fill ``slug_id`` and ``part_number_mpn`` when Gemini or the URL provides hints.

    Args:
        specs: Structured output from Gemini (may omit identity fields).
        url: Product page URL used for MPN extraction and slug fallback.

    Returns:
        Same instance or a copy with identity fields populated when inferable.
    """
    updates: dict[str, str] = {}
    if not specs.part_number_mpn:
        if mpn := extract_mpn_from_url(url):
            updates["part_number_mpn"] = mpn
    if not specs.slug_id:
        base = specs.model_name or updates.get("part_number_mpn") or extract_mpn_from_url(url)
        if base:
            updates["slug_id"] = _slugify_identity(base)
    if not updates:
        return specs
    return specs.model_copy(update=updates)


async def check_cache_node(state: AgentState) -> dict:
    """LangGraph node: resolve specs from SQLite via MCP (URL then MPN alias).

    Lookup order: ``url_mapping`` by URL, then ``mpn_mapping`` using MPN from the
    URL, then ``laptops`` by ``slug_id``. MCP failures are logged and treated as
    a cache miss (empty partial update).

    Args:
        state: Requires ``state['url']``; skipped when ``error`` is already set.

    Returns:
        Partial update with ``specs`` on hit and cleared ``error``; empty dict on
        miss or failure.
    """
    if state.get("error"):
        system_log.debug("check_cache_node skipped: prior error=%s", state["error"])
        return {}

    url = state["url"]
    _emit_status(_NODE_STATUS["check_cache"], node="check_cache", phase="cache")
    user_log.info("Checking spec cache for URL.")
    system_log.info("check_cache_node start url=%s", url)

    try:
        slug_id = await lookup_slug_by_url(url)
        if not slug_id:
            extracted_mpn = extract_mpn_from_url(url)
            if extracted_mpn:
                system_log.info(
                    "check_cache_node url miss; trying mpn=%s",
                    extracted_mpn,
                )
                slug_id = await lookup_slug_by_mpn(extracted_mpn)

        if not slug_id:
            _emit_status("Cache miss; scraping product page.", node="check_cache", phase="cache")
            user_log.info("Cache miss for URL.")
            system_log.info("check_cache_node miss url=%s", url)
            return {}

        specs = await fetch_laptop_by_slug(slug_id)
        if not specs:
            system_log.warning(
                "check_cache_node slug=%s missing laptops row",
                slug_id,
            )
            return {}

        _emit_status(
            f"Cache hit for {specs.model_name or slug_id}.",
            node="check_cache",
            phase="cache",
        )
        user_log.info("Cache hit: slug_id=%s", slug_id)
        system_log.info("check_cache_node hit slug_id=%s", slug_id)
        return {"specs": specs, "error": ""}
    except Exception as exc:
        system_log.warning("check_cache_node failed (continuing without cache): %s", exc)
        return {}


def route_after_check_cache(state: AgentState) -> str:
    """Conditional edge after cache lookup: scrape or skip to fit evaluation.

    Args:
        state: Post-``check_cache`` state; may include ``specs`` on hit.

    Returns:
        ``"evaluate_use_case"`` when ``specs`` is set; otherwise ``"extract_page"``.
    """
    if state.get("specs"):
        system_log.info("route_after_check_cache -> evaluate_use_case (cache hit)")
        return "evaluate_use_case"
    return "extract_page"


async def save_to_cache_node(state: AgentState) -> dict:
    """LangGraph node: persist specs and URL/MPN mappings to SQLite via MCP.

    Runs after each successful ``parse_specs`` on the scrape path. Failures are
    logged and do not set ``state['error']``.

    Args:
        state: Requires ``specs`` with ``slug_id`` and ``url``.

    Returns:
        Empty dict (cache writes are side effects only).
    """
    if state.get("error"):
        system_log.debug("save_to_cache_node skipped: error=%s", state["error"])
        return {}

    specs = state.get("specs")
    if not specs or not specs.slug_id:
        system_log.debug("save_to_cache_node skipped: no slug_id")
        return {}

    url = state["url"]
    _emit_status(_NODE_STATUS["save_to_cache"], node="save_to_cache", phase="cache")
    user_log.info("Saving specs to cache (slug_id=%s).", specs.slug_id)
    system_log.info("save_to_cache_node start slug_id=%s url=%s", specs.slug_id, url)

    try:
        await save_laptop_cache(url=url, specs=specs)
        _emit_status("Specs saved to cache.", node="save_to_cache", phase="cache")
        system_log.info("save_to_cache_node success slug_id=%s", specs.slug_id)
        return {}
    except Exception as exc:
        system_log.warning("save_to_cache_node failed (non-fatal): %s", exc)
        return {}


def _format_search_query(model_name: str | None, field_name: str) -> str:
    """Build a DuckDuckGo query for one missing spec field.

    Args:
        model_name: Laptop model from parsed specs, or ``None`` for generic ``laptop``.
        field_name: ``LaptopSpecs`` attribute name (e.g. ``npu_tops``).

    Returns:
        Query string such as ``"Dell XPS 13 NPU TOPS specs"``.
    """
    label = _FIELD_SEARCH_LABELS.get(field_name, field_name.replace("_", " "))
    product = (model_name or "laptop").strip()
    return f"{product} {label} specs"


async def extract_page_node(state: AgentState) -> dict:
    """LangGraph node: scrape product page text into ``raw_content``.

    Args:
        state: Current graph state; uses ``state['url']``.

    Returns:
        Partial update with ``raw_content`` on success, or ``error`` message.
        Empty dict if a prior error already exists.
    """
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
    """LangGraph node: extract ``LaptopSpecs`` from ``raw_content`` via Gemini.

    Args:
        state: Must include non-empty ``raw_content`` unless ``error`` is set.

    Returns:
        Partial update with ``specs`` on success, or ``error`` on failure.
        Empty dict if skipped due to prior error.
    """
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
                    content=build_extract_specs_prompt(
                        raw_content[:_CONTENT_LIMIT],
                    ),
                ),
            ],
        )
        specs = _ensure_identity_fields(specs, state["url"])
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
    """LangGraph node: web-search each null spec field and enrich ``raw_content``.

    Args:
        state: Requires ``specs`` with ``unknown_fields``; reads ``retry_count``.

    Returns:
        Partial update with longer ``raw_content``, incremented ``retry_count``,
        and cleared ``error`` on success. Per-field search failures are appended
        as text rather than aborting the whole node.
    """
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
    """Conditional edge: search loop vs use-case evaluation.

    Args:
        state: Post-parse state including ``specs`` and ``retry_count``.

    Returns:
        ``"search_missing_specs"`` when null fields remain and retries < 2;
        otherwise ``"evaluate_use_case"``.
    """
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
    """LangGraph node: judge fit between ``specs`` and ``use_case``.

    Args:
        state: Needs ``use_case`` and ``specs``; no-ops if ``error`` already set.

    Returns:
        Partial update with ``fit_evaluation`` (recommendation + rationale),
        or ``error`` if evaluation cannot run or Gemini fails.
    """
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
                    content=build_fit_evaluation_prompt(
                        use_case=use_case,
                        specs_json=specs.model_dump_json(indent=2),
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
    """Compile the laptop spec LangGraph with cache, scrape, parse, search, and fit.

    Flow: ``check_cache`` → (hit) ``evaluate_use_case`` | (miss) ``extract_page`` →
    ``parse_specs`` → ``save_to_cache`` → search loop or ``evaluate_use_case``.

    Returns:
        Compiled graph runnable via ``ainvoke`` / ``astream``. Wrap runs in
        ``mcp_sqlite_session()`` so cache nodes can reach SQLite.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("extract_page", extract_page_node)
    workflow.add_node("parse_specs", parse_specs_node)
    workflow.add_node("save_to_cache", save_to_cache_node)
    workflow.add_node("search_missing_specs", search_missing_specs_node)
    workflow.add_node("evaluate_use_case", evaluate_use_case_node)
    workflow.add_edge(START, "check_cache")
    workflow.add_conditional_edges(
        "check_cache",
        route_after_check_cache,
        {
            "extract_page": "extract_page",
            "evaluate_use_case": "evaluate_use_case",
        },
    )
    workflow.add_edge("extract_page", "parse_specs")
    workflow.add_edge("parse_specs", "save_to_cache")
    workflow.add_conditional_edges(
        "save_to_cache",
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
    """Convert a custom stream chunk to ``AgentStatus``.

    Args:
        payload: Dict from ``get_stream_writer()`` or a plain string.

    Returns:
        ``AgentStatus`` when payload is usable; otherwise ``None``.
    """
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
    """Convert an ``updates`` stream chunk for one node to ``AgentStatus``.

    Args:
        node: LangGraph node name (e.g. ``parse_specs``).
        update: State partial returned by that node, or ``None``.

    Returns:
        ``AgentStatus`` for errors, parse summaries, or fit results; ``None`` if
        nothing user-facing should be logged.
    """
    if not update:
        return None

    if update.get("error"):
        return {"message": f"Error: {update['error']}", "node": node, "phase": "error"}

    if node == "check_cache" and update.get("specs"):
        specs = update["specs"]
        if isinstance(specs, LaptopSpecs):
            return {
                "message": f"Loaded specs from cache ({specs.model_name or specs.slug_id}).",
                "node": node,
                "phase": "cache",
            }

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
    """Write a status event to user and system loggers.

    Args:
        status: Event from streaming helpers.

    Returns:
        None.
    """
    user_log.info("%s", status["message"])
    system_log.debug(
        "stream status node=%s phase=%s message=%s",
        status.get("node"),
        status.get("phase"),
        status["message"],
    )


async def stream_agent(url: str, use_case: str) -> AsyncIterator[AgentStatus]:
    """Run the graph and yield progress events for each ``astream`` chunk.

    Args:
        url: Product page URL.
        use_case: Validated user use-case description.

    Yields:
        ``AgentStatus`` events during execution; final yield has ``phase='done'``
        and ``final_state`` set.

    Returns:
        Nothing directly; consumers read the last yielded ``final_state``.
    """
    graph = build_graph()
    final_state: AgentState | None = None

    _log_status(
        {"message": f"Starting agent for {url}", "phase": "start"},
    )
    _log_status({"message": f"Use case: {use_case}", "phase": "start"})
    system_log.info("stream_agent start url=%s use_case=%s", url, use_case)

    try:
        async with mcp_sqlite_session():
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
    """Execute the full agent pipeline and return the terminal state.

    Logs progress to the console via ``stream_agent``.

    Args:
        url: Product page URL.
        use_case: User use-case string.

    Returns:
        Final ``AgentState`` including ``specs`` and ``fit_evaluation`` when
        successful.
    """
    result: AgentState | None = None
    async for status in stream_agent(url, use_case):
        if status.get("phase") == "done":
            result = status.get("final_state")
    return result if result is not None else _initial_state(url, use_case)


def _print_fit_evaluation(evaluation: UseCaseFitEvaluation) -> None:
    """Print human-readable and JSON fit evaluation to stdout.

    Args:
        evaluation: Structured fit result from ``evaluate_use_case_node``.

    Returns:
        None.
    """
    label = _RECOMMENDATION_LABELS.get(
        evaluation.recommendation,
        evaluation.recommendation,
    )
    print("\n--- Use case fit ---")
    print(f"Recommendation: {label}")
    print(f"Rationale: {evaluation.rationale}")
    print(evaluation.model_dump_json(indent=2))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CLI entrypoint.

    Returns:
        Namespace with ``url`` and optional ``use_case`` attributes.
    """
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
    """Print an error to stderr and terminate the process.

    Args:
        message: User-facing error text.
        code: Unix exit code (e.g. 1 runtime, 2 validation, 130 interrupt).

    Returns:
        Does not return; calls ``sys.exit``.
    """
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
