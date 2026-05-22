import asyncio
import os
import time

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from errors import (
    AgentError,
    SearchError,
    empty_page_content_error,
    wrap_exception,
)
from logging_config import get_system_logger, get_user_logger

user_log = get_user_logger()
system_log = get_system_logger()

_DEFAULT_TIMEOUT_MS = 60_000
_DEFAULT_WAIT_UNTIL = "domcontentloaded"
_MIN_PAGE_CHARS = 100

_ddg_search = DuckDuckGoSearchRun()


@tool
async def search_web(query: str) -> str:
    """Search the web for laptop specification information."""
    cleaned = (query or "").strip()
    if not cleaned:
        raise SearchError(
            "Search query is empty.",
            hint="Internal error: search was invoked without a query string.",
        )

    user_log.info("Searching the web…")
    system_log.info("search_web start query=%s", cleaned)
    try:
        results = await asyncio.to_thread(_ddg_search.invoke, cleaned)
        if not results or not str(results).strip():
            raise SearchError(
                f'No search results returned for "{cleaned}".',
                hint="Try a more specific query or retry later.",
                code="search_empty",
            )
        system_log.info(
            "search_web success query=%s result_chars=%d",
            cleaned,
            len(results),
        )
        return results
    except (SearchError, AgentError):
        raise
    except Exception as exc:
        system_log.exception("search_web failed query=%s", cleaned)
        raise wrap_exception(exc, phase="search", context={"query": cleaned}) from exc


@tool
async def fetch_page_text(url: str) -> str:
    """Load a retail product page and return visible body text."""
    timeout_ms = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS)))
    wait_until = os.getenv("PLAYWRIGHT_WAIT_UNTIL", _DEFAULT_WAIT_UNTIL)

    user_log.info("Fetching product page…")
    system_log.info(
        "fetch_page_text start url=%s wait_until=%s timeout_ms=%s",
        url,
        wait_until,
        timeout_ms,
    )
    started = time.perf_counter()

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                text = await page.evaluate("() => document.body.innerText")
            finally:
                await browser.close()
                system_log.debug("Browser closed for url=%s", url)
    except PlaywrightError as exc:
        system_log.exception(
            "fetch_page_text playwright error url=%s elapsed_s=%.2f",
            url,
            time.perf_counter() - started,
        )
        raise wrap_exception(exc, phase="scrape", context={"url": url}) from exc
    except Exception as exc:
        system_log.exception(
            "fetch_page_text failed url=%s elapsed_s=%.2f",
            url,
            time.perf_counter() - started,
        )
        raise wrap_exception(exc, phase="scrape", context={"url": url}) from exc

    elapsed = time.perf_counter() - started
    char_count = len(text or "")
    if char_count < _MIN_PAGE_CHARS:
        raise empty_page_content_error()

    user_log.info("Page loaded (%s characters).", f"{char_count:,}")
    system_log.info(
        "fetch_page_text success url=%s chars=%d elapsed_s=%.2f",
        url,
        char_count,
        elapsed,
    )
    return text
