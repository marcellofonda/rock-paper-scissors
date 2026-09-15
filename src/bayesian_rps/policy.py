"""Turn predictive probabilities into a move."""

from __future__ import annotations

import random

from .game import MOVES, PAYOFF_MATRIX, Move
from .model import Probabilities


def expected_utilities(prediction: Probabilities) -> Probabilities:
    """Compute expected payoff for playing Rock, Paper and Scissors."""
    return tuple(
        sum(PAYOFF_MATRIX[action][opponent] * prediction[opponent] for opponent in MOVES)
        for action in MOVES
    )  # type: ignore[return-value]


def choose_move(prediction: Probabilities, rng: random.Random) -> Move:
    """Choose uniformly among the moves with maximum expected payoff."""
    utilities = expected_utilities(prediction)
    best_value = max(utilities)
    best_moves = [move for move in MOVES if utilities[move] == best_value]
    return rng.choice(best_moves)
