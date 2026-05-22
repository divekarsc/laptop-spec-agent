"""LLM prompt for evaluating laptop fit against a user use case."""

FIT_EVALUATION_TEMPLATE = """You are advising a laptop buyer. Given structured hardware specs and the user's use case, decide how well this laptop fits.

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


def build_fit_evaluation_prompt(*, use_case: str, specs_json: str) -> str:
    """Format the use-case fit evaluation prompt.

    Args:
        use_case: User's natural-language use case (≤25 words).
        specs_json: JSON serialization of ``LaptopSpecs``.

    Returns:
        Complete prompt string for Gemini structured fit evaluation.
    """
    return FIT_EVALUATION_TEMPLATE.format(use_case=use_case, specs=specs_json)
