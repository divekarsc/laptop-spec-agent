"""Fast manufacturer part number (MPN) extraction from product URLs."""

import re
from urllib.parse import parse_qs, urlparse

_MPN_QUERY_KEYS = frozenset(
    {
        "mpn",
        "part_number",
        "partnumber",
        "sku",
        "model",
        "modelnumber",
        "pn",
        "productid",
    }
)

_PATH_MPN_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-_.]{4,31}$", re.IGNORECASE)


def extract_mpn_from_url(url: str) -> str | None:
    """Extract a likely MPN or SKU token from a retail product URL.

    Checks common query-parameter names first, then scans path segments from
    the end for alphanumeric product codes.

    Args:
        url: Product page URL.

    Returns:
        Extracted MPN/SKU string, or ``None`` when nothing plausible is found.
    """
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return None

    for key, values in parse_qs(parsed.query, keep_blank_values=False).items():
        if key.lower() in _MPN_QUERY_KEYS and values:
            candidate = values[0].strip()
            if candidate:
                return candidate

    for segment in reversed(parsed.path.split("/")):
        segment = segment.strip()
        if not segment or segment.lower() in {"product", "p", "dp", "laptop", "laptops"}:
            continue
        if _PATH_MPN_RE.fullmatch(segment):
            return segment

    return None
