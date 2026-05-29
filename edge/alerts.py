"""Alert delivery for identified edges (console, file, or future integrations)."""


def print_edges(edges) -> None:
    """Pretty-print edge bets to stdout."""
    raise NotImplementedError


def save_edges(edges, output_path: str) -> None:
    """Write edge bets to a CSV/JSON file for review."""
    raise NotImplementedError
