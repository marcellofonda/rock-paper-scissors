"""A neutral match engine that works with any two players."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .game import Move, payoff
from .players import Player


@dataclass(frozen=True)
class MatchRound:
    round_number: int
    player_one_move: Move
    player_two_move: Move
    player_one_payoff: int


@dataclass(frozen=True)
class MatchSummary:
    rounds: int
    player_one_wins: int
    player_two_wins: int
    draws: int
    player_one_mean_payoff: float
    player_one_move_counts: tuple[int, int, int]
    player_two_move_counts: tuple[int, int, int]


def summarize_match(records: list[MatchRound]) -> MatchSummary:
    """Aggregate outcomes and move frequencies for a completed match."""
    if not records:
        raise ValueError("cannot summarize a match with no rounds")

    player_one_wins = sum(record.player_one_payoff == 1 for record in records)
    player_two_wins = sum(record.player_one_payoff == -1 for record in records)
    draws = len(records) - player_one_wins - player_two_wins
    moves_one = tuple(
        sum(record.player_one_move == move for record in records) for move in Move
    )
    moves_two = tuple(
        sum(record.player_two_move == move for record in records) for move in Move
    )

    return MatchSummary(
        rounds=len(records),
        player_one_wins=player_one_wins,
        player_two_wins=player_two_wins,
        draws=draws,
        player_one_mean_payoff=sum(r.player_one_payoff for r in records) / len(records),
        player_one_move_counts=moves_one,  # type: ignore[arg-type]
        player_two_move_counts=moves_two,  # type: ignore[arg-type]
    )


def play_match(
    player_one: Player,
    player_two: Player,
    rounds: int,
    on_round: Callable[[MatchRound], None] | None = None,
) -> list[MatchRound]:
    """Play simultaneous rounds and show each player the result afterwards."""
    records = []

    for round_number in range(1, rounds + 1):
        move_one = player_one.choose_move()
        move_two = player_two.choose_move()
        result = payoff(move_one, move_two)

        player_one.observe(move_one, move_two, result)
        player_two.observe(move_two, move_one, -result)
        record = MatchRound(round_number, move_one, move_two, result)
        records.append(record)
        if on_round is not None:
            on_round(record)

    return records
