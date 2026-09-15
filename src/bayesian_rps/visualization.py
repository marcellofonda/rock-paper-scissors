"""Plots for a single simulated game."""

from __future__ import annotations

import math
from pathlib import Path

from .game import Move
from .match import MatchRound
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


def plot_match(
    records: list[MatchRound],
    output: str | Path,
    player_one_log_losses: list[float] | None = None,
    player_two_log_losses: list[float] | None = None,
) -> Path:
    """Plot score, move frequencies and optional Bayesian predictive quality."""
    import matplotlib.pyplot as plt

    rounds = [record.round_number for record in records]
    cumulative_score = []
    score = 0
    for record in records:
        score += record.player_one_payoff
        cumulative_score.append(score)

    has_log_loss = player_one_log_losses is not None or player_two_log_losses is not None
    row_count = 4 if has_log_loss else 3
    figure, axes = plt.subplots(row_count, 1, figsize=(10, 3 * row_count), sharex=True)

    axes[0].plot(rounds, cumulative_score, color="tab:green")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Player 1\ncumulative payoff")

    colors = ("tab:red", "tab:blue", "tab:orange")
    for axis, attribute, title in (
        (axes[1], "player_one_move", "Player 1 move frequency"),
        (axes[2], "player_two_move", "Player 2 move frequency"),
    ):
        for move, color in zip(Move, colors):
            count = 0
            frequencies = []
            for round_number, record in enumerate(records, start=1):
                count += getattr(record, attribute) == move
                frequencies.append(count / round_number)
            axis.plot(rounds, frequencies, label=move.name.title(), color=color)
        axis.set_ylabel(title)
        axis.set_ylim(0, 1)
        axis.legend(ncol=3)

    if has_log_loss:
        log_loss_axis = axes[3]
        for losses, label, color in (
            (player_one_log_losses, "Player 1", "tab:purple"),
            (player_two_log_losses, "Player 2", "tab:brown"),
        ):
            if losses is None:
                continue
            cumulative_loss = 0.0
            running_mean = []
            for round_number, loss in enumerate(losses, start=1):
                cumulative_loss += loss
                running_mean.append(cumulative_loss / round_number)
            log_loss_axis.plot(rounds, running_mean, label=label, color=color)
        log_loss_axis.axhline(-math.log(1 / 3), color="black", linestyle="--")
        log_loss_axis.set_ylabel("Mean log loss")
        log_loss_axis.legend()

    axes[-1].set_xlabel("Round")
    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
