"""Rules of Rock-Paper-Scissors."""

from __future__ import annotations

from enum import IntEnum


class Move(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


MOVES = tuple(Move)

# Rows are our move; columns are the opponent's move.
PAYOFF_MATRIX = (
    (0, -1, 1),
    (1, 0, -1),
    (-1, 1, 0),
)


def payoff(our_move: Move, opponent_move: Move) -> int:
    """Return +1 for a win, 0 for a draw and -1 for a loss."""
    return PAYOFF_MATRIX[our_move][opponent_move]
