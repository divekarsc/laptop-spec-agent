"""LangGraph state definition for the laptop spec agent."""

from typing import NotRequired, TypedDict

from schema import LaptopSpecs, UseCaseFitEvaluation


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes.

    Required keys are set at graph start; optional keys are filled as nodes run.

    Attributes:
        url: Retail product page URL to scrape.
        use_case: User's natural-language laptop use case (max 25 words).
        raw_content: Visible text from the product page plus appended search snippets.
        specs: Structured hardware fields extracted by Gemini.
        retry_count: Number of web-search enrichment rounds completed (max 2).
        fit_evaluation: Gemini assessment of specs vs use_case.
        error: Human-readable failure message; non-empty stops productive work.
    """

    url: str
    use_case: str
    raw_content: NotRequired[str]
    specs: NotRequired[LaptopSpecs]
    retry_count: NotRequired[int]
    fit_evaluation: NotRequired[UseCaseFitEvaluation]
    error: NotRequired[str]
