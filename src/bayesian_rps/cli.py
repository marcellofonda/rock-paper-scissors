"""Small command-line entry point for presentation-ready experiments."""

import argparse
from pathlib import Path

from .opponents import (
    PreviousMoveCounterOpponent,
    RepeatingMarkovOpponent,
    RockBiasedOpponent,
    UniformRandomOpponent,
    WinStayLoseShiftOpponent,
)
from .simulation import simulate
from .visualization import plot_game


OPPONENTS = {
    "uniform-random": UniformRandomOpponent,
    "rock-biased": RockBiasedOpponent,
    "repeating-markov": RepeatingMarkovOpponent,
    "previous-move-counter": PreviousMoveCounterOpponent,
    "win-stay-lose-shift": WinStayLoseShiftOpponent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a Bayesian RPS player")
    parser.add_argument(
        "--opponent", choices=OPPONENTS, default="previous-move-counter"
    )
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        help="Output image path (generated automatically when omitted)",
    )
    args = parser.parse_args()

    records = simulate(OPPONENTS[args.opponent](), args.rounds, args.seed)
    mean_payoff = sum(record.payoff for record in records) / len(records)
    mean_log_loss = sum(record.log_loss for record in records) / len(records)
    opponent_name = args.opponent.replace("-", "_")
    default_name = f"game_{opponent_name}_{args.rounds}_{args.seed}.png"
    output = Path(args.output) if args.output else Path("results") / default_name
    image_path = plot_game(records, output)

    print(f"Opponent:       {args.opponent}")
    print(f"Rounds:         {args.rounds}")
    print(f"Mean payoff:    {mean_payoff:.3f}")
    print(f"Mean log loss:  {mean_log_loss:.3f}")
    print(f"Plot saved to:  {image_path}")


if __name__ == "__main__":
    main()
