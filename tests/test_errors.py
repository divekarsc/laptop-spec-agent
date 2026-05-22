import pytest

from errors import (
    LLMError,
    ScrapeError,
    ValidationError,
    error_message_for_state,
    validate_product_url,
    validate_use_case,
    wrap_exception,
)


class TestValidateProductUrl:
    def test_accepts_https_url(self) -> None:
        assert validate_product_url("  https://shop.example/laptop  ") == (
            "https://shop.example/laptop"
        )

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError, match="required"):
            validate_product_url("")

    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(ValidationError, match="http:// or https://"):
            validate_product_url("ftp://files.example/laptop")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError, match="too long"):
            validate_product_url("https://x.com/" + "a" * 2048)


class TestValidateUseCase:
    def test_normalizes_whitespace(self) -> None:
        assert validate_use_case("  gaming   and   school  ") == "gaming and school"

    def test_accepts_25_words(self) -> None:
        words = " ".join(f"w{i}" for i in range(25))
        assert len(validate_use_case(words).split()) == 25

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError, match="required"):
            validate_use_case("   ")

    def test_rejects_over_25_words(self) -> None:
        words = " ".join(f"w{i}" for i in range(26))
        with pytest.raises(ValidationError, match="25 words"):
            validate_use_case(words)


class TestWrapException:
    def test_playwright_timeout_maps_to_scrape_error(self) -> None:
        exc = Exception("Page.goto: Timeout 30000ms exceeded")
        err = wrap_exception(exc, phase="extract_page", context={"url": "https://x.com"})
        assert isinstance(err, ScrapeError)
        assert err.code == "scrape_timeout"
        assert "https://x.com" in err.user_message

    def test_gemini_quota_maps_to_llm_error(self) -> None:
        err = wrap_exception(
            Exception("429 resource_exhausted quota"),
            phase="parse_specs",
        )
        assert isinstance(err, LLMError)
        assert err.code == "llm_quota_error"

    def test_passes_through_agent_error(self) -> None:
        original = ValidationError("bad input")
        assert wrap_exception(original, phase="cli") is original

    def test_error_message_for_state_includes_hint(self) -> None:
        msg = error_message_for_state(
            Exception("invalid api key"),
            phase="parse_specs",
        )
        assert "Gemini API key" in msg
