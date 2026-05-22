"""Legacy entrypoint; use ``graph.py`` as the primary CLI."""


def main() -> None:
    """Print a placeholder message.

    The real CLI lives in ``graph.py`` (``uv run python graph.py <url>``).

    Returns:
        None. Writes one line to stdout.
    """
    print("Hello from laptop-spec-agent!")


if __name__ == "__main__":
    main()
