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
bayesian-rps --opponent previous-move-counter --rounds 500 --seed 0
```

The command prints the main metrics and saves a plot with a descriptive name,
such as `results/game_previous_move_counter_500_0.png`.

The contextual model is used by default. Select the global baseline with:

```powershell
bayesian-rps --model global --opponent rock-biased --rounds 500 --seed 0
```

The global model learns one distribution `P(next opponent move)`. The
contextual model instead learns a separate distribution for each previous pair
of moves, `P(next opponent move | previous pair)`. Comparing them shows whether
the extra context is informative enough to compensate for having fewer
observations in each context.

## Virtual opponents

- `uniform-random`: chooses Rock, Paper and Scissors independently with equal
  probability. This is the control: there is no exploitable pattern to learn.
- `rock-biased`: chooses `(Rock, Paper, Scissors)` with probabilities
  `(0.60, 0.25, 0.15)`. It has a simple global bias but ignores past rounds.
- `repeating-markov`: repeats its own previous move with probability `0.70`;
  otherwise it chooses randomly. Its next move depends on its own last move.
- `previous-move-counter`: with probability `0.80`, plays the move that beats
  the player's previous move. It is a directly reactive opponent.
- `win-stay-lose-shift`: repeats a move after winning; after a draw or loss, it
  moves cyclically to its next action.

You can still choose a custom destination with `--output`:

```powershell
bayesian-rps --opponent previous-move-counter --output results/my_experiment.png
```

## Run the tests

```powershell
python -m pytest
```

## Play any two players

The same match engine supports humans, Bayesian players and predefined automatic
strategies. For example, compare the two Bayesian models:

```powershell
bayesian-rps-play bayesian-contextual bayesian-global --rounds 500 --seed 0
```

Play against the contextual model:

```powershell
bayesian-rps-play human bayesian-contextual --rounds 20
```

Or play human versus human:

```powershell
bayesian-rps-play human human --rounds 10
```

Human moves are hidden while they are entered, so the second player cannot see
the first player's choice. Both moves are revealed after the round. At the end,
the CLI reports win/draw/loss rates, mean payoff and move frequencies. For each
Bayesian player it also reports mean predictive log loss. A match plot is saved
with a name such as
`results/match_bayesian_contextual_vs_bayesian_global_500_0.png`. It shows
cumulative payoff, cumulative move frequencies and, when applicable, running
mean log loss. Use `--output` to choose a different path.
