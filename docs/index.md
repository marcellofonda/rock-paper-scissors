---
icon: lucide/rocket
---

# Bayesian Rock-Paper-Scissors 🗿🧻✂️

This project is a Bayesian glimpse at the classic rock-paper-scissors (RPS) game. 

RPS has a long history of ruining friendships, dating back to the Han dynasty (206 BCE – 220 CE) according to Xie Zhaozhe, and is spread all around the world.

!!! info
    To learn more about the game, visit [its Wikipedia page](https://en.wikipedia.org/wiki/Rock_paper_scissors): it is a very interesting reference and it offers an entertaining read about the game, its origins, and variations.

The aim of this project is to develop a very simple bot that can learn the strategy used by another player. The key assumption will be that each move the opponent makes is based only on the moves played during the previous round. The learning will be achieved through Bayesian inference.

## The game

At each round the two players simultaneously choose between rock (R), paper (P), or scissors (S). Famously, paper beats rock, which beats scissors, which beats paper.

``` mermaid
graph RL
  B[Scissors] -->|beats| C[Paper];
  C -->|beats| D[Rock];
  D -->|beats| B;
```

If both players make the same choice, then it's a draw. If scores are kept, then we can add 1 for each victory, 0 for each tie and -1 for each loss. Therefore, a *payoff matrix* can be defined:


||R|P|S|
|-|--|--|--|
|R|0|-1|1|
|P|1|0|-1|
|S|-1|1|0|

The rows of the payoff matrix represent the move played by our bot, while the
columns represent the opponent's move.

## Learning from the previous round

A completely random opponent cannot be systematically exploited. Human and
automatic players, however, may exhibit patterns. For example, an opponent
might tend to play Paper after seeing us play Rock.

We assume that the opponent's next move depends only on the moves played during
the previous round.

Let

\[
M_t \in \{R, P, S\}
\]

be the move played by our bot at round \(t\), and let

\[
O_t \in \{R, P, S\}
\]

be the opponent's move.
The bot's objective is to estimate the distribution

\[
P(O_{t+1} \mid M_t, O_t)
\]

and play its move in order to maximize the expected payoff. This is done learning the distribution from the data, according to the rules of bayesian statistics.
