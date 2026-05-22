"""LLM setup for structured spec extraction via Google Gemini."""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from errors import ConfigurationError, LLMError, missing_api_key_error, wrap_exception
from logging_config import get_system_logger, get_user_logger
from schema import LaptopSpecs, UseCaseFitEvaluation

user_log = get_user_logger()
system_log = get_system_logger()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_structured_llm = None
_fit_evaluation_llm = None


def _resolve_google_api_key() -> str:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            system_log.debug("Using API key from %s", name)
            return value
    raise missing_api_key_error()


def get_gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def build_structured_llm():
    """Build the structured-output Gemini runnable; raises ``ConfigurationError`` if misconfigured."""
    try:
        api_key = _resolve_google_api_key()
        model = get_gemini_model_name()
        user_log.info("Using Gemini model: %s", model)
        system_log.info("build_structured_llm model=%s", model)

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key,
        )
        return llm.with_structured_output(LaptopSpecs)
    except ConfigurationError:
        raise
    except Exception as exc:
        system_log.exception("build_structured_llm failed")
        raise wrap_exception(exc, phase="config") from exc


def get_structured_llm():
    """Lazy singleton for the structured LLM."""
    global _structured_llm
    if _structured_llm is None:
        _structured_llm = build_structured_llm()
    return _structured_llm


def build_fit_evaluation_llm():
    """Structured LLM for use-case fit assessment."""
    try:
        api_key = _resolve_google_api_key()
        model = get_gemini_model_name()
        system_log.info("build_fit_evaluation_llm model=%s", model)
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key,
        )
        return llm.with_structured_output(UseCaseFitEvaluation)
    except ConfigurationError:
        raise
    except Exception as exc:
        system_log.exception("build_fit_evaluation_llm failed")
        raise wrap_exception(exc, phase="config") from exc


def get_fit_evaluation_llm():
    """Lazy singleton for fit evaluation."""
    global _fit_evaluation_llm
    if _fit_evaluation_llm is None:
        _fit_evaluation_llm = build_fit_evaluation_llm()
    return _fit_evaluation_llm
