"""Convert model output to implied probability and calculate edge vs. book."""


def american_to_prob(odds: int) -> float:
    """Convert American odds (e.g. -110, +150) to implied probability."""
    raise NotImplementedError


def remove_vig(prob_home: float, prob_away: float) -> tuple[float, float]:
    """Strip vig from raw implied probabilities using the additive method."""
    raise NotImplementedError


def calc_edge(model_prob: float, fair_prob: float) -> float:
    """Edge = model_prob - fair_prob. Positive means value bet."""
    raise NotImplementedError


def kelly_stake(edge: float, odds: int, bankroll: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake in dollars."""
    raise NotImplementedError


def find_edges(predictions, odds_db_path: str, min_edge: float = 0.03):
    """Cross-reference model predictions against current odds. Return value bets."""
    raise NotImplementedError
