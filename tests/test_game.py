from bayesian_rps.game import Move, payoff


def test_payoff_rules():
    assert payoff(Move.ROCK, Move.SCISSORS) == 1
    assert payoff(Move.ROCK, Move.PAPER) == -1
    assert payoff(Move.ROCK, Move.ROCK) == 0

