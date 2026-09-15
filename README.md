# Bayesian Rock–Paper–Scissors

A small example of sequential Bayesian learning.

At every round, the player:

1. observes the previous pair of moves `(player, opponent)`;
2. predicts the opponent's next move with a Dirichlet-multinomial model;
3. computes the expected payoff of Rock, Paper and Scissors;
4. plays a move with maximum expected payoff;
5. observes the result and updates the posterior.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run an experiment

```powershell
bayesian-rps --opponent reactive --rounds 500 --seed 0
```

Available opponents are `random`, `biased`, `markov`, `reactive`, and
`win-stay-lose-shift`. The command prints the main metrics and saves a plot with
a descriptive name such as `results/game_reactive_500_0.png`.

You can still choose a custom destination with `--output`:

```powershell
bayesian-rps --opponent reactive --output results/my_experiment.png
```

## Run the tests

```powershell
python -m pytest
```
