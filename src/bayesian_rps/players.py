"""A common interface for Bayesian, automatic and human players."""

from __future__ import annotations

import getpass
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .game import Move
from .model import Context, ContextualDirichletModel
from .opponents import Opponent
from .policy import choose_move
from .simulation import PredictiveModel


class Player(Protocol):
    """Anything that can participate in a match."""

    name: str

    def choose_move(self) -> Move: ...

    def observe(self, own_move: Move, opponent_move: Move, result: int) -> None: ...


@dataclass
class BayesianPlayer:
    """A player that learns the opponent's transitions online."""

    model: PredictiveModel = field(default_factory=ContextualDirichletModel)
    seed: int = 0
    name: str = "Bayesian player"
    _previous_context: Context | None = field(default=None, init=False)
    _rng: random.Random = field(init=False)
    _current_prediction: tuple[float, float, float] | None = field(
        default=None, init=False
    )
    log_losses: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_move(self) -> Move:
        if self._previous_context is None:
            total = sum(self.model.alpha)
            prediction = tuple(value / total for value in self.model.alpha)
        else:
            prediction = self.model.predict(self._previous_context)
        self._current_prediction = prediction
        return choose_move(prediction, self._rng)

    def observe(self, own_move: Move, opponent_move: Move, result: int) -> None:
        if self._current_prediction is not None:
            self.log_losses.append(-math.log(self._current_prediction[opponent_move]))
        if self._previous_context is not None:
            self.model.update(self._previous_context, opponent_move)
        self._previous_context = (own_move, opponent_move)

    @property
    def mean_log_loss(self) -> float | None:
        if not self.log_losses:
            return None
        return sum(self.log_losses) / len(self.log_losses)


@dataclass
class AutomaticPlayer:
    """Adapt one of the predefined virtual opponents to the Player interface."""

    strategy: Opponent
    seed: int = 0
    name: str = "Automatic player"
    _previous_own_move: Move | None = field(default=None, init=False)
    _previous_opponent_move: Move | None = field(default=None, init=False)
    _previous_result: int | None = field(default=None, init=False)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_move(self) -> Move:
        return self.strategy.choose_move(
            self._previous_opponent_move,
            self._previous_own_move,
            -self._previous_result if self._previous_result is not None else None,
            self._rng,
        )

    def observe(self, own_move: Move, opponent_move: Move, result: int) -> None:
        self._previous_own_move = own_move
        self._previous_opponent_move = opponent_move
        self._previous_result = result


@dataclass
class HumanPlayer:
    """Read moves from the terminal; hidden input permits human-vs-human play."""

    name: str = "Human"
    hidden_input: bool = True
    input_function: Callable[[str], str] | None = None

    def choose_move(self) -> Move:
        read = self.input_function
        if read is None:
            read = getpass.getpass if self.hidden_input else input

        aliases = {
            "r": Move.ROCK,
            "rock": Move.ROCK,
            "p": Move.PAPER,
            "paper": Move.PAPER,
            "s": Move.SCISSORS,
            "scissors": Move.SCISSORS,
        }
        while True:
            answer = read(f"{self.name}, choose [R]ock, [P]aper or [S]cissors: ")
            move = aliases.get(answer.strip().lower())
            if move is not None:
                return move
            print("Please enter R, P or S.")

    def observe(self, own_move: Move, opponent_move: Move, result: int) -> None:
        pass
