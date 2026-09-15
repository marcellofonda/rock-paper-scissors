"""Plots for a single simulated game."""

from __future__ import annotations

import math
from pathlib import Path

from .simulation import RoundRecord


def plot_game(records: list[RoundRecord], output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    rounds = [record.round_number for record in records]
    cumulative_payoff = []
    running_log_loss = []
    payoff_total = 0
    loss_total = 0.0

    for record in records:
        payoff_total += record.payoff
        loss_total += record.log_loss
        cumulative_payoff.append(payoff_total)
        running_log_loss.append(loss_total / record.round_number)

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ("Rock", "Paper", "Scissors")
    for move, label in enumerate(labels):
        axes[0].plot(rounds, [r.prediction[move] for r in records], label=label)
    axes[0].set_ylabel("Predictive probability")
    axes[0].set_ylim(0, 1)
    axes[0].legend(ncol=3)

    axes[1].plot(rounds, cumulative_payoff, color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Cumulative payoff")

    axes[2].plot(rounds, running_log_loss, color="tab:red")
    axes[2].axhline(-math.log(1 / 3), color="black", linestyle="--")
    axes[2].set_ylabel("Mean log loss")
    axes[2].set_xlabel("Round")

    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
