"""Run a sequential game and record everything needed for analysis."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from .game import Move, payoff
from .model import Context, ContextualDirichletModel, Probabilities
from .opponents import Opponent
from .policy import choose_move, expected_utilities


@dataclass(frozen=True)
class RoundRecord:
    round_number: int
    player_move: Move
    opponent_move: Move
    prediction: Probabilities
    expected_payoffs: Probabilities
    payoff: int
    log_loss: float


class PredictiveModel(Protocol):
    alpha: Probabilities

    def predict(self, context: Context) -> Probabilities: ...

    def update(self, context: Context, observed_move: Move) -> None: ...


def simulate(
    opponent: Opponent,
    rounds: int = 500,
    seed: int = 0,
    alpha: Probabilities = (1.0, 1.0, 1.0),
    model: PredictiveModel | None = None,
) -> list[RoundRecord]:
    """Play a complete game using one reproducible random generator."""
    rng = random.Random(seed)
    if model is None:
        model = ContextualDirichletModel(alpha)
    records: list[RoundRecord] = []
    previous_context: Context | None = None
    previous_payoff: int | None = None

    for round_number in range(1, rounds + 1):
        prediction = (
            model.predict(previous_context)
            if previous_context is not None
            else tuple(value / sum(alpha) for value in alpha)
        )
        utilities = expected_utilities(prediction)
        player_move = choose_move(prediction, rng)

        previous_player = previous_context[0] if previous_context else None
        previous_opponent = previous_context[1] if previous_context else None
        opponent_move = opponent.choose_move(
            previous_player, previous_opponent, previous_payoff, rng
        )

        round_payoff = payoff(player_move, opponent_move)
        log_loss = -math.log(prediction[opponent_move])

        records.append(
            RoundRecord(
                round_number,
                player_move,
                opponent_move,
                prediction,
                utilities,
                round_payoff,
                log_loss,
            )
        )

        if previous_context is not None:
            model.update(previous_context, opponent_move)

        previous_context = (player_move, opponent_move)
        previous_payoff = round_payoff

    return records
