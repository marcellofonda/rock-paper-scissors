"""Interactive and model-vs-model matches."""

from __future__ import annotations

import argparse
from pathlib import Path

from .match import MatchRound, play_match, summarize_match
from .model import ContextualDirichletModel, GlobalDirichletModel
from .opponents import (
    PreviousMoveCounterOpponent,
    RepeatingMarkovOpponent,
    RockBiasedOpponent,
    UniformRandomOpponent,
    WinStayLoseShiftOpponent,
)
from .players import AutomaticPlayer, BayesianPlayer, HumanPlayer, Player
from .visualization import plot_match


PLAYER_TYPES = (
    "human",
    "bayesian-contextual",
    "bayesian-global",
    "uniform-random",
    "rock-biased",
    "repeating-markov",
    "previous-move-counter",
    "win-stay-lose-shift",
)


def make_player(kind: str, name: str, seed: int) -> Player:
    if kind == "human":
        return HumanPlayer(name=name)
    if kind == "bayesian-contextual":
        return BayesianPlayer(ContextualDirichletModel(), seed, name)
    if kind == "bayesian-global":
        return BayesianPlayer(GlobalDirichletModel(), seed, name)

    strategies = {
        "uniform-random": UniformRandomOpponent,
        "rock-biased": RockBiasedOpponent,
        "repeating-markov": RepeatingMarkovOpponent,
        "previous-move-counter": PreviousMoveCounterOpponent,
        "win-stay-lose-shift": WinStayLoseShiftOpponent,
    }
    return AutomaticPlayer(strategies[kind](), seed, name)


def print_round(record: MatchRound) -> None:
    print(
        f"Round {record.round_number}: "
        f"{record.player_one_move.name} vs {record.player_two_move.name} "
        f"({record.player_one_payoff:+d})"
    )


def format_move_frequencies(counts: tuple[int, int, int], rounds: int) -> str:
    rock, paper, scissors = (100 * count / rounds for count in counts)
    return f"R {rock:5.1f}% | P {paper:5.1f}% | S {scissors:5.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a match between any two players")
    parser.add_argument("player_one", choices=PLAYER_TYPES)
    parser.add_argument("player_two", choices=PLAYER_TYPES)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        help="Output image path (generated automatically when omitted)",
    )
    args = parser.parse_args()

    player_one = make_player(args.player_one, "Player 1", args.seed)
    player_two = make_player(args.player_two, "Player 2", args.seed + 1)
    records = play_match(player_one, player_two, args.rounds, on_round=print_round)

    summary = summarize_match(records)
    print("\nMatch summary")
    print(f"Player 1 wins:  {summary.player_one_wins:4d} "
          f"({100 * summary.player_one_wins / summary.rounds:5.1f}%)")
    print(f"Player 2 wins:  {summary.player_two_wins:4d} "
          f"({100 * summary.player_two_wins / summary.rounds:5.1f}%)")
    print(f"Draws:          {summary.draws:4d} "
          f"({100 * summary.draws / summary.rounds:5.1f}%)")
    print(f"Player 1 mean payoff: {summary.player_one_mean_payoff:+.3f}")
    print(
        "Player 1 moves: "
        + format_move_frequencies(summary.player_one_move_counts, summary.rounds)
    )
    print(
        "Player 2 moves: "
        + format_move_frequencies(summary.player_two_move_counts, summary.rounds)
    )

    if isinstance(player_one, BayesianPlayer):
        print(f"Player 1 mean log loss: {player_one.mean_log_loss:.3f}")
    if isinstance(player_two, BayesianPlayer):
        print(f"Player 2 mean log loss: {player_two.mean_log_loss:.3f}")

    player_one_name = args.player_one.replace("-", "_")
    player_two_name = args.player_two.replace("-", "_")
    default_name = (
        f"match_{player_one_name}_vs_{player_two_name}_{args.rounds}_{args.seed}.png"
    )
    output = Path(args.output) if args.output else Path("results") / default_name
    image_path = plot_match(
        records,
        output,
        player_one.log_losses if isinstance(player_one, BayesianPlayer) else None,
        player_two.log_losses if isinstance(player_two, BayesianPlayer) else None,
    )
    print(f"Plot saved to: {image_path}")


if __name__ == "__main__":
    main()
