"""Dirichlet-multinomial model for the opponent's next move."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .game import MOVES, Move

Context = Tuple[Move, Move]
Probabilities = Tuple[float, float, float]


@dataclass
class ContextualDirichletModel:
    """Learn P(next opponent move | previous pair of moves)."""

    alpha: Probabilities = (1.0, 1.0, 1.0)
    counts: list[list[list[int]]] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.alpha) != 3 or any(value <= 0 for value in self.alpha):
            raise ValueError("alpha must contain three positive values")
        self.counts = [[[0 for _ in MOVES] for _ in MOVES] for _ in MOVES]

    def predict(self, context: Context) -> Probabilities:
        """Return the posterior predictive probabilities for R, P and S."""
        our_previous, opponent_previous = context
        observations = self.counts[our_previous][opponent_previous]
        posterior = [prior + count for prior, count in zip(self.alpha, observations)]
        total = sum(posterior)
        return tuple(value / total for value in posterior)  # type: ignore[return-value]

    def update(self, context: Context, observed_move: Move) -> None:
        """Add one observed transition to the relevant context."""
        our_previous, opponent_previous = context
        self.counts[our_previous][opponent_previous][observed_move] += 1
