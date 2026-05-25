"""MCP SQLite client for the laptop spec cache (Cross-Retailer Identity Aliasing)."""

from __future__ import annotations

import ast
import json
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from logging_config import get_system_logger
from schema import LaptopSpecs

system_log = get_system_logger()

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "laptop_cache.db"

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS laptops (
        slug_id TEXT PRIMARY KEY,
        model_name TEXT,
        processor_architecture TEXT,
        npu_tops REAL,
        display_panel_type TEXT,
        thermal_design_power_watts INTEGER,
        usb_c_gan_charging_support INTEGER,
        refresh_rate_hz INTEGER,
        weight_kg REAL,
        screen_size_inches REAL,
        battery_capacity_wh INTEGER,
        operating_system TEXT,
        gpu_type TEXT,
        unknown_fields_json TEXT
    )
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS url_mapping (
        url TEXT PRIMARY KEY,
        slug_id TEXT NOT NULL,
        FOREIGN KEY(slug_id) REFERENCES laptops(slug_id)
    )
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS mpn_mapping (
        part_number_mpn TEXT PRIMARY KEY,
        slug_id TEXT NOT NULL,
        FOREIGN KEY(slug_id) REFERENCES laptops(slug_id)
    )
    """.strip(),
)

_active_session: ContextVar[ClientSession | None] = ContextVar(
    "mcp_sqlite_session",
    default=None,
)


def sql_literal(value: str) -> str:
    """Return a single-quoted SQLite string literal with escaped quotes.

    Args:
        value: Raw string to embed in SQL (URLs, slugs, text fields).

    Returns:
        Safe SQL literal, e.g. ``'it''s'`` for input ``it's``.
    """
    return "'" + value.replace("'", "''") + "'"


def sql_optional_str(value: str | None) -> str:
    """Format an optional TEXT column value for SQL.

    Args:
        value: Column text, or ``None`` for SQL ``NULL``.

    Returns:
        ``NULL`` or a quoted literal from ``sql_literal``.
    """
    if value is None:
        return "NULL"
    return sql_literal(value)


def sql_optional_float(value: float | None) -> str:
    """Format an optional REAL column value for SQL.

    Args:
        value: Numeric column value, or ``None`` for SQL ``NULL``.

    Returns:
        ``NULL`` or a float literal string.
    """
    if value is None:
        return "NULL"
    return repr(float(value))


def sql_optional_int(value: int | None) -> str:
    """Format an optional INTEGER column value for SQL.

    Args:
        value: Integer column value, or ``None`` for SQL ``NULL``.

    Returns:
        ``NULL`` or a decimal integer string.
    """
    if value is None:
        return "NULL"
    return str(int(value))


def sql_optional_bool(value: bool | None) -> str:
    """Format an optional BOOLEAN column as SQLite 0/1.

    Args:
        value: Boolean column value, or ``None`` for SQL ``NULL``.

    Returns:
        ``NULL``, ``'1'``, or ``'0'``.
    """
    if value is None:
        return "NULL"
    return "1" if value else "0"


def _db_path() -> Path:
    """Resolve the SQLite database file path from environment or default.

    Returns:
        Absolute path to ``SQLITE_DB_PATH`` or ``data/laptop_cache.db``.
    """
    raw = os.environ.get("SQLITE_DB_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_DB_PATH.resolve()


def _stdio_params() -> StdioServerParameters:
    """Build stdio launch parameters for the MCP SQLite server.

    Reads ``MCP_SQLITE_COMMAND`` and ``MCP_SQLITE_ARGS`` when set; otherwise
    uses ``uvx mcp-server-sqlite --db-path <db>``.

    Returns:
        Parameters passed to ``stdio_client``.
    """
    command = os.environ.get("MCP_SQLITE_COMMAND", "uvx").strip() or "uvx"
    db_path = str(_db_path())
    default_args = ["mcp-server-sqlite", "--db-path", db_path]
    extra = os.environ.get("MCP_SQLITE_ARGS", "").strip()
    args = extra.split() if extra else default_args
    return StdioServerParameters(command=command, args=args)


def _tool_text(result: CallToolResult) -> str:
    """Extract the first text payload from an MCP tool result.

    Args:
        result: Response from ``call_tool``.

    Returns:
        Text body, or empty string when no text content is present.
    """
    if not result.content:
        return ""
    first = result.content[0]
    return getattr(first, "text", "") or ""


def parse_read_rows(result: CallToolResult) -> list[dict[str, Any]]:
    """Parse rows returned by the MCP ``read_query`` tool.

    Args:
        result: Tool response whose text is a Python list of row dicts or JSON.

    Returns:
        List of row mappings; empty when the tool returns no rows.
    """
    text = _tool_text(result).strip()
    if not text:
        return []
    try:
        data = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def row_to_laptop_specs(row: dict[str, Any]) -> LaptopSpecs:
    """Map a ``laptops`` table row to ``LaptopSpecs``.

    Converts database column names (e.g. ``thermal_design_power_watts``) to
    Pydantic field names (e.g. ``tdp_watts``). ``part_number_mpn`` is not stored
    on the laptops row and is left ``None``.

    Args:
        row: Single row dict from ``SELECT * FROM laptops``.

    Returns:
        Populated ``LaptopSpecs`` instance.
    """
    tdp = row.get("thermal_design_power_watts")
    gan = row.get("usb_c_gan_charging_support")
    npu = row.get("npu_tops")
    refresh = row.get("refresh_rate_hz")
    weight = row.get("weight_kg")
    screen = row.get("screen_size_inches")
    battery = row.get("battery_capacity_wh")
    gpu_type = row.get("gpu_type")

    return LaptopSpecs(
        slug_id=row.get("slug_id"),
        model_name=row.get("model_name"),
        processor_architecture=row.get("processor_architecture"),
        npu_tops=float(npu) if npu is not None else None,
        display_panel_type=row.get("display_panel_type"),
        tdp_watts=float(tdp) if tdp is not None else None,
        gan_charging_support=bool(gan) if gan is not None else None,
        refresh_rate_hz=int(refresh) if refresh is not None else None,
        weight_kg=float(weight) if weight is not None else None,
        screen_size_inches=float(screen) if screen is not None else None,
        battery_capacity_wh=int(battery) if battery is not None else None,
        operating_system=row.get("operating_system"),
        gpu_type=gpu_type if gpu_type in ("dedicated", "integrated", "hybrid") else None,
        part_number_mpn=None,
    )


def laptop_specs_to_insert_sql(specs: LaptopSpecs) -> str:
    """Build ``INSERT OR REPLACE`` SQL for the ``laptops`` table.

    Args:
        specs: Parsed specs including ``slug_id`` and ``unknown_fields``.

    Returns:
        Single SQL statement with safely escaped string literals.
    """
    unknown_json = json.dumps(specs.unknown_fields)
    tdp_int: int | None = None
    if specs.tdp_watts is not None:
        tdp_int = int(round(specs.tdp_watts))

    columns = (
        "slug_id",
        "model_name",
        "processor_architecture",
        "npu_tops",
        "display_panel_type",
        "thermal_design_power_watts",
        "usb_c_gan_charging_support",
        "refresh_rate_hz",
        "weight_kg",
        "screen_size_inches",
        "battery_capacity_wh",
        "operating_system",
        "gpu_type",
        "unknown_fields_json",
    )
    values = (
        sql_literal(specs.slug_id or ""),
        sql_optional_str(specs.model_name),
        sql_optional_str(specs.processor_architecture),
        sql_optional_float(specs.npu_tops),
        sql_optional_str(specs.display_panel_type),
        sql_optional_int(tdp_int),
        sql_optional_bool(specs.gan_charging_support),
        sql_optional_int(specs.refresh_rate_hz),
        sql_optional_float(specs.weight_kg),
        sql_optional_float(specs.screen_size_inches),
        sql_optional_int(specs.battery_capacity_wh),
        sql_optional_str(specs.operating_system),
        sql_optional_str(specs.gpu_type),
        sql_literal(unknown_json),
    )
    return (
        f"INSERT OR REPLACE INTO laptops ({', '.join(columns)}) "
        f"VALUES ({', '.join(values)});"
    )


async def _call_write(session: ClientSession, query: str) -> None:
    """Execute a mutating SQL statement via the MCP ``write_query`` tool.

    Args:
        session: Active MCP client session.
        query: SQL for INSERT, UPDATE, DELETE, or DDL.

    Returns:
        None.
    """
    await session.call_tool("write_query", {"query": query})


async def _call_read(session: ClientSession, query: str) -> list[dict[str, Any]]:
    """Execute a SELECT via the MCP ``read_query`` tool and parse rows.

    Args:
        session: Active MCP client session.
        query: Read-only SQL (typically SELECT).

    Returns:
        Parsed row dicts from the tool response.
    """
    result = await session.call_tool("read_query", {"query": query})
    return parse_read_rows(result)


async def init_schema(session: ClientSession) -> None:
    """Create cache tables if they do not exist.

    Args:
        session: Active MCP client session.

    Returns:
        None. Logs the resolved database path on success.
    """
    for statement in _SCHEMA_STATEMENTS:
        await _call_write(session, statement)
    system_log.info("SQLite cache schema initialized db=%s", _db_path())


def get_session() -> ClientSession | None:
    """Return the MCP session bound to the current async context, if any.

    Returns:
        Active ``ClientSession`` while inside ``mcp_sqlite_session``; else ``None``.
    """
    return _active_session.get()


@asynccontextmanager
async def mcp_sqlite_session() -> AsyncIterator[ClientSession]:
    """Open an MCP stdio session to ``mcp-server-sqlite`` and initialize schema.

    Creates the database parent directory, starts the server subprocess, runs
    ``init_schema``, and stores the session in a context variable for cache helpers.

    Yields:
        Initialized ``ClientSession`` for the graph run duration.

    Returns:
        None (async context manager).
    """
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    params = _stdio_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await init_schema(session)
            token = _active_session.set(session)
            try:
                yield session
            finally:
                _active_session.reset(token)


async def lookup_slug_by_url(url: str) -> str | None:
    """Return ``slug_id`` for a cached product URL, if present.

    Args:
        url: Retail product page URL (exact match on ``url_mapping.url``).

    Returns:
        ``slug_id`` string when mapped; ``None`` on miss or when no MCP session.
    """
    session = get_session()
    if session is None:
        return None
    rows = await _call_read(
        session,
        f"SELECT slug_id FROM url_mapping WHERE url = {sql_literal(url)};",
    )
    if not rows:
        return None
    slug = rows[0].get("slug_id")
    return str(slug) if slug else None


async def lookup_slug_by_mpn(part_number_mpn: str) -> str | None:
    """Return ``slug_id`` for a cached manufacturer part number, if present.

    Args:
        part_number_mpn: MPN or SKU token.

    Returns:
        ``slug_id`` string when mapped; ``None`` on miss or when no MCP session.
    """
    session = get_session()
    if session is None:
        return None
    rows = await _call_read(
        session,
        "SELECT slug_id FROM mpn_mapping WHERE part_number_mpn = "
        f"{sql_literal(part_number_mpn)};",
    )
    if not rows:
        return None
    slug = rows[0].get("slug_id")
    return str(slug) if slug else None


async def fetch_laptop_by_slug(slug_id: str) -> LaptopSpecs | None:
    """Load cached hardware specs for a ``slug_id``.

    Args:
        slug_id: Primary key in the ``laptops`` table.

    Returns:
        ``LaptopSpecs`` when a row exists; ``None`` if missing or no MCP session.
    """
    session = get_session()
    if session is None:
        return None
    rows = await _call_read(
        session,
        f"SELECT * FROM laptops WHERE slug_id = {sql_literal(slug_id)};",
    )
    if not rows:
        return None
    return row_to_laptop_specs(rows[0])


async def save_laptop_cache(
    *,
    url: str,
    specs: LaptopSpecs,
) -> None:
    """Persist specs and URL/MPN alias mappings.

    Writes ``INSERT OR REPLACE`` to ``laptops`` and ``INSERT OR IGNORE`` to
    ``url_mapping`` and ``mpn_mapping`` (when ``part_number_mpn`` is set).

    Args:
        url: Product page URL for this run.
        specs: Extracted specs; must include ``slug_id`` or the call is a no-op.

    Returns:
        None. Requires an active session from ``mcp_sqlite_session``.
    """
    session = get_session()
    if session is None or not specs.slug_id:
        return

    await _call_write(session, laptop_specs_to_insert_sql(specs))
    await _call_write(
        session,
        "INSERT OR IGNORE INTO url_mapping (url, slug_id) VALUES "
        f"({sql_literal(url)}, {sql_literal(specs.slug_id)});",
    )
    if specs.part_number_mpn:
        await _call_write(
            session,
            "INSERT OR IGNORE INTO mpn_mapping (part_number_mpn, slug_id) VALUES "
            f"({sql_literal(specs.part_number_mpn)}, {sql_literal(specs.slug_id)});",
        )
