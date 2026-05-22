"""User-facing and system error types with actionable messages."""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base error with a message safe to show in CLI and graph state."""

    code: str = "agent_error"

    def __init__(
        self,
        user_message: str,
        *,
        code: str | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.user_message = user_message
        self.hint = hint
        self.cause = cause
        if code:
            self.code = code
        detail = user_message if not hint else f"{user_message} {hint}"
        super().__init__(detail)

    def formatted(self) -> str:
        if self.hint and self.hint not in self.user_message:
            return f"{self.user_message} Hint: {self.hint}"
        return self.user_message


class ConfigurationError(AgentError):
    code = "configuration_error"


class ValidationError(AgentError):
    code = "validation_error"


class ScrapeError(AgentError):
    code = "scrape_error"


class LLMError(AgentError):
    code = "llm_error"


class SearchError(AgentError):
    code = "search_error"


_EXIT_HINT = "See logs/system.log for full stack traces."


MAX_USE_CASE_WORDS = 25


def validate_use_case(text: str) -> str:
    """Return normalized use case text or raise ``ValidationError``."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise ValidationError(
            "Use case description is required.",
            hint="Describe how you will use the laptop in 25 words or fewer.",
        )
    words = cleaned.split()
    if len(words) > MAX_USE_CASE_WORDS:
        raise ValidationError(
            f"Use case must be at most {MAX_USE_CASE_WORDS} words (got {len(words)}).",
            hint="Shorten your description and try again.",
        )
    return cleaned


def validate_product_url(url: str) -> str:
    """Return a normalized URL or raise ``ValidationError``."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValidationError(
            "Product URL is required.",
            hint='Pass a URL: uv run python graph.py "https://..."',
        )
    if not cleaned.startswith(("http://", "https://")):
        raise ValidationError(
            "Product URL must start with http:// or https://.",
            hint=f"Received: {cleaned[:80]}{'…' if len(cleaned) > 80 else ''}",
        )
    if len(cleaned) > 2048:
        raise ValidationError(
            "Product URL is too long (max 2048 characters).",
        )
    return cleaned


def _match_google_api_error(message: str) -> LLMError | None:
    lower = message.lower()
    if "api key" in lower or "api_key" in lower or "invalid key" in lower:
        return LLMError(
            "Gemini API key was rejected.",
            hint=(
                "Check GOOGLE_API_KEY in .env "
                "(https://aistudio.google.com/apikey)."
            ),
            code="llm_auth_error",
        )
    if "quota" in lower or "rate" in lower or "429" in lower or "resource_exhausted" in lower:
        return LLMError(
            "Gemini API quota or rate limit exceeded.",
            hint="Wait and retry, or switch GEMINI_MODEL to a lighter model.",
            code="llm_quota_error",
        )
    if "not found" in lower and "model" in lower:
        return LLMError(
            "Configured Gemini model was not found.",
            hint="Set GEMINI_MODEL to a valid model id (e.g. gemini-2.5-flash).",
            code="llm_model_error",
        )
    return None


def _match_playwright_error(exc: BaseException) -> ScrapeError:
    message = str(exc)
    lower = message.lower()

    if "timeout" in lower and "goto" in lower:
        return ScrapeError(
            f"Timed out loading the product page ({exc.__class__.__name__}).",
            hint=(
                "Try PLAYWRIGHT_TIMEOUT_MS=90000 or PLAYWRIGHT_WAIT_UNTIL=load "
                "in .env, or use a simpler product URL."
            ),
            cause=exc if isinstance(exc, Exception) else None,
            code="scrape_timeout",
        )

    if "net::" in lower or "ns_error" in lower or "connection" in lower:
        return ScrapeError(
            "Network error while loading the product page.",
            hint="Check the URL, VPN, and firewall; confirm the site is reachable in a browser.",
            cause=exc if isinstance(exc, Exception) else None,
            code="scrape_network",
        )

    if "executable doesn't exist" in lower or "playwright install" in lower:
        return ScrapeError(
            "Playwright browser is not installed.",
            hint="Run: uv run playwright install chromium",
            cause=exc if isinstance(exc, Exception) else None,
            code="scrape_browser_missing",
        )

    return ScrapeError(
        "Failed to scrape the product page.",
        hint=_EXIT_HINT,
        cause=exc if isinstance(exc, Exception) else None,
    )


def _match_search_error(exc: BaseException, *, query: str | None = None) -> SearchError:
    message = str(exc)
    prefix = f'Web search failed for "{query}". ' if query else "Web search failed. "
    if "ddgs" in message.lower() or "duckduckgo" in message.lower():
        return SearchError(
            prefix + "DuckDuckGo search backend is unavailable.",
            hint="Install deps with uv sync; check network connectivity.",
            cause=exc if isinstance(exc, Exception) else None,
        )
    return SearchError(
        prefix.strip(),
        hint=_EXIT_HINT,
        cause=exc if isinstance(exc, Exception) else None,
    )


def wrap_exception(
    exc: BaseException,
    *,
    phase: str,
    context: dict[str, Any] | None = None,
) -> AgentError:
    """Map a raw exception to an ``AgentError`` with a useful user message."""
    if isinstance(exc, AgentError):
        return exc

    ctx = context or {}
    query = ctx.get("query")
    url = ctx.get("url")

    if phase in ("scrape", "extract_page"):
        err = _match_playwright_error(exc)
        if url:
            err.user_message = f"{err.user_message} URL: {url}"
        return err

    if phase in ("search", "search_missing_specs"):
        return _match_search_error(exc, query=query)

    if phase in ("llm", "parse_specs", "extract"):
        if mapped := _match_google_api_error(str(exc)):
            return mapped
        return LLMError(
            "Gemini failed to extract structured specs.",
            hint=_EXIT_HINT,
            cause=exc if isinstance(exc, Exception) else None,
        )

    if phase == "config":
        return ConfigurationError(
            str(exc),
            hint=_EXIT_HINT,
            cause=exc if isinstance(exc, Exception) else None,
        )

    return AgentError(
        f"Unexpected error during {phase}.",
        hint=_EXIT_HINT,
        cause=exc if isinstance(exc, Exception) else None,
        code="unexpected_error",
    )


def error_message_for_state(exc: BaseException, *, phase: str, **context: Any) -> str:
    """Return a single string suitable for ``AgentState['error']``."""
    return wrap_exception(exc, phase=phase, context=context).formatted()


def cli_exit_message(exc: BaseException) -> str:
    """Format an exception for stderr when the CLI exits."""
    if isinstance(exc, AgentError):
        return exc.formatted()
    if isinstance(exc, KeyboardInterrupt):
        return "Interrupted."
    wrapped = wrap_exception(exc, phase="cli")
    return wrapped.formatted()


def missing_api_key_error() -> ConfigurationError:
    return ConfigurationError(
        "No Google API key found.",
        hint=(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env "
            "(https://aistudio.google.com/apikey)."
        ),
        code="missing_api_key",
    )


def empty_page_content_error() -> ValidationError:
    return ValidationError(
        "The scraped page contained no readable text.",
        hint="The site may block bots, require login, or load content via JavaScript only.",
        code="empty_content",
    )


def summarize_run_failure(state: dict[str, Any]) -> str:
    """Build a CLI message when the agent ends in a failed state."""
    if error := state.get("error"):
        return str(error)
    if not state.get("specs"):
        return (
            "Agent finished without extracted specs. "
            "The page may lack spec details or the model could not parse them."
        )
    specs = state["specs"]
    if hasattr(specs, "unknown_fields") and specs.unknown_fields:
        return (
            f"Partial specs only; still missing: {', '.join(specs.unknown_fields)}. "
            f"Try another URL or increase search retries."
        )
    return "Unknown failure."
