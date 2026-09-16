"""Run the reproducible experiments used by the documentation.

Run this file from the repository root after installing the package:

    python experiments/run_experiments.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from bayesian_rps.model import ContextualDirichletModel, GlobalDirichletModel
from bayesian_rps.opponents import (
    Opponent,
    PreviousMoveCounterOpponent,
    RepeatingMarkovOpponent,
    RockBiasedOpponent,
    UniformRandomOpponent,
    WinStayLoseShiftOpponent,
)
from bayesian_rps.simulation import PredictiveModel, simulate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results"
DEFAULT_FIGURES_DIR = ROOT / "docs" / "images" / "experiments"


@dataclass(frozen=True)
class Experiment:
    slug: str
    label: str
    opponent_factory: Callable[[], Opponent]
    include_in_learning_curves: bool = True


@dataclass(frozen=True)
class RunResult:
    experiment: str
    experiment_label: str
    model: str
    seed: int
    mean_payoff: float
    mean_log_loss: float
    running_payoff: tuple[float, ...]


EXPERIMENTS = (
    Experiment("uniform-random", "Uniform random", UniformRandomOpponent),
    Experiment("rock-biased", "Rock biased", RockBiasedOpponent),
    Experiment("repeating-markov", "Repeating Markov", RepeatingMarkovOpponent),
    Experiment(
        "previous-move-counter",
        "Previous-move counter",
        PreviousMoveCounterOpponent,
    ),
    Experiment(
        "win-stay-lose-shift",
        "Win-stay, lose-shift",
        WinStayLoseShiftOpponent,
        include_in_learning_curves=False,
    ),
)

MODELS: tuple[tuple[str, Callable[[], PredictiveModel]], ...] = (
    ("global", GlobalDirichletModel),
    ("contextual", ContextualDirichletModel),
)


def run_all(rounds: int, seeds: int) -> list[RunResult]:
    results = []
    for experiment in EXPERIMENTS:
        for model_name, model_factory in MODELS:
            for seed in range(seeds):
                records = simulate(
                    experiment.opponent_factory(),
                    rounds=rounds,
                    seed=seed,
                    model=model_factory(),
                )
                payoff_total = 0
                running_payoff = []
                for round_number, record in enumerate(records, start=1):
                    payoff_total += record.payoff
                    running_payoff.append(payoff_total / round_number)
                results.append(
                    RunResult(
                        experiment.slug,
                        experiment.label,
                        model_name,
                        seed,
                        payoff_total / rounds,
                        statistics.fmean(record.log_loss for record in records),
                        tuple(running_payoff),
                    )
                )
    return results


def confidence_interval(values: Iterable[float]) -> tuple[float, float, float, float]:
    observations = tuple(values)
    mean = statistics.fmean(observations)
    if len(observations) == 1:
        return mean, 0.0, mean, mean
    standard_deviation = statistics.stdev(observations)
    half_width = 1.96 * standard_deviation / math.sqrt(len(observations))
    return mean, standard_deviation, mean - half_width, mean + half_width


def grouped_results(
    results: Iterable[RunResult], experiment: str, model: str
) -> list[RunResult]:
    return [
        result
        for result in results
        if result.experiment == experiment and result.model == model
    ]


def write_summary(results: list[RunResult], path: Path, rounds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "opponent",
                "model",
                "rounds",
                "seeds",
                "mean_payoff",
                "payoff_std",
                "payoff_ci95_low",
                "payoff_ci95_high",
                "mean_log_loss",
                "log_loss_std",
                "log_loss_ci95_low",
                "log_loss_ci95_high",
            )
        )
        for experiment in EXPERIMENTS:
            for model_name, _ in MODELS:
                group = grouped_results(results, experiment.slug, model_name)
                payoff = confidence_interval(result.mean_payoff for result in group)
                log_loss = confidence_interval(result.mean_log_loss for result in group)
                writer.writerow(
                    (
                        experiment.slug,
                        model_name,
                        rounds,
                        len(group),
                        *(f"{value:.6f}" for value in payoff),
                        *(f"{value:.6f}" for value in log_loss),
                    )
                )


def plot_payoff_comparison(results: list[RunResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.5))
    x_positions = list(range(len(EXPERIMENTS)))
    width = 0.34
    colors = {"global": "#6c757d", "contextual": "#2a9d8f"}

    for offset, (model_name, _) in zip((-width / 2, width / 2), MODELS):
        means = []
        errors = []
        for experiment in EXPERIMENTS:
            group = grouped_results(results, experiment.slug, model_name)
            mean, _, low, high = confidence_interval(
                result.mean_payoff for result in group
            )
            means.append(mean)
            errors.append((high - low) / 2)
        axis.bar(
            [position + offset for position in x_positions],
            means,
            width,
            yerr=errors,
            capsize=3,
            label=model_name.title(),
            color=colors[model_name],
        )

    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions, [item.label for item in EXPERIMENTS], rotation=15)
    axis.set_ylabel("Mean payoff per round")
    axis.set_ylim(-0.4, 1.08)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_learning_curves(results: list[RunResult], path: Path, rounds: int) -> None:
    import matplotlib.pyplot as plt

    experiments = [item for item in EXPERIMENTS if item.include_in_learning_curves]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    colors = {"global": "#6c757d", "contextual": "#2a9d8f"}
    round_numbers = list(range(1, rounds + 1))

    for axis, experiment in zip(axes.flat, experiments):
        for model_name, _ in MODELS:
            group = grouped_results(results, experiment.slug, model_name)
            means = []
            lows = []
            highs = []
            for index in range(rounds):
                mean, _, low, high = confidence_interval(
                    result.running_payoff[index] for result in group
                )
                means.append(mean)
                lows.append(low)
                highs.append(high)
            axis.plot(
                round_numbers,
                means,
                label=model_name.title(),
                color=colors[model_name],
            )
            axis.fill_between(
                round_numbers,
                lows,
                highs,
                color=colors[model_name],
                alpha=0.15,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(experiment.label)
        axis.grid(alpha=0.2)

    axes[0, 0].legend()
    figure.supxlabel("Round")
    figure.supylabel("Running mean payoff")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the aggregate results used in the documentation."
    )
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.seeds < 1:
        raise SystemExit("--rounds and --seeds must both be positive")

    results = run_all(args.rounds, args.seeds)
    summary_path = args.results_dir / "summary.csv"
    payoff_path = args.figures_dir / "payoff-comparison.png"
    learning_path = args.figures_dir / "learning-curves.png"
    write_summary(results, summary_path, args.rounds)
    plot_payoff_comparison(results, payoff_path)
    plot_learning_curves(results, learning_path, args.rounds)

    print(f"Completed {len(results)} simulations")
    print(f"Summary:         {summary_path}")
    print(f"Payoff figure:   {payoff_path}")
    print(f"Learning curves: {learning_path}")


if __name__ == "__main__":
    main()
