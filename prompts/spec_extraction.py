"""LLM prompt for extracting ``LaptopSpecs`` from retail page text."""

EXTRACT_SPECS_TEMPLATE = """Extract laptop hardware specifications from the retail product page text below.
Use only information present in the text; leave fields null when not stated or unclear.

Page text:
{content}"""


def build_extract_specs_prompt(content: str) -> str:
    """Format the spec-extraction prompt with scraped page text.

    Args:
        content: Truncated ``raw_content`` from the product page (and search appendices).

    Returns:
        Complete prompt string for Gemini structured extraction.
    """
    return EXTRACT_SPECS_TEMPLATE.format(content=content)
