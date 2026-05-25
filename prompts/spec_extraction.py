"""LLM prompt for extracting ``LaptopSpecs`` from retail page text."""

EXTRACT_SPECS_TEMPLATE = """Extract laptop hardware specifications from the retail product page text below.
Use only information present in the text; leave fields null when not stated or unclear.

Include when stated:
- weight_kg: total laptop weight in kilograms (convert from pounds if needed).
- screen_size_inches: diagonal display size in inches.
- battery_capacity_wh: battery capacity in watt-hours (Wh).
- operating_system: preinstalled OS name/version.
- gpu_type: one of dedicated, integrated, or hybrid (switchable discrete + integrated).

Also set:
- slug_id: a stable lowercase slug (letters, digits, hyphens) from the model name or MPN, e.g. "dell-xps-13-9340".
- part_number_mpn: manufacturer part number or SKU when present in the text (not the retailer item ID unless it is the MPN).

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
