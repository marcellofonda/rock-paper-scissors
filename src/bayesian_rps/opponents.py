"""Simple virtual opponents used in the experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .game import MOVES, Move


class Opponent(Protocol):
    def choose_move(
        self,
        previous_player_move: Move | None,
        previous_opponent_move: Move | None,
        previous_payoff: int | None,
        rng: random.Random,
    ) -> Move: ...


@dataclass(frozen=True)
class UniformRandomOpponent:
    """Choose every move independently with equal probability."""

    def choose_move(self, previous_player_move, previous_opponent_move, previous_payoff, rng):
        return rng.choice(MOVES)


@dataclass(frozen=True)
class RockBiasedOpponent:
    """Choose Rock more often, independently of previous rounds."""

    probabilities: tuple[float, float, float] = (0.60, 0.25, 0.15)

    def choose_move(self, previous_player_move, previous_opponent_move, previous_payoff, rng):
        return rng.choices(MOVES, weights=self.probabilities, k=1)[0]


@dataclass(frozen=True)
class PreviousMoveCounterOpponent:
    """Usually play the move that beats the player's previous move."""

    reaction_probability: float = 0.80

    def choose_move(self, previous_player_move, previous_opponent_move, previous_payoff, rng):
        if previous_player_move is None or rng.random() > self.reaction_probability:
            return rng.choice(MOVES)
        return Move((previous_player_move + 1) % 3)


@dataclass(frozen=True)
class RepeatingMarkovOpponent:
    """Repeat the previous move with a configurable probability."""

    repeat_probability: float = 0.70

    def choose_move(self, previous_player_move, previous_opponent_move, previous_payoff, rng):
        if previous_opponent_move is None or rng.random() > self.repeat_probability:
            return rng.choice(MOVES)
        return previous_opponent_move


@dataclass(frozen=True)
class WinStayLoseShiftOpponent:
    """Repeat after a win; otherwise move cyclically to the next action."""

    def choose_move(self, previous_player_move, previous_opponent_move, previous_payoff, rng):
        if previous_opponent_move is None or previous_payoff is None:
            return rng.choice(MOVES)
        opponent_won = previous_payoff == -1
        if opponent_won:
            return previous_opponent_move
        return Move((previous_opponent_move + 1) % 3)
