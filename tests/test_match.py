import pytest

from bayesian_rps.match import play_match, summarize_match
from bayesian_rps.model import ContextualDirichletModel, GlobalDirichletModel
from bayesian_rps.players import BayesianPlayer, HumanPlayer


def test_two_bayesian_players_can_play_each_other():
    contextual = BayesianPlayer(ContextualDirichletModel(), seed=1)
    global_player = BayesianPlayer(GlobalDirichletModel(), seed=2)

    records = play_match(contextual, global_player, rounds=50)

    assert len(records) == 50
    contextual_observations = sum(
        count
        for our_move in contextual.model.counts
        for opponent_move in our_move
        for count in opponent_move
    )
    assert contextual_observations == 49
    assert sum(global_player.model.counts) == 49
    assert contextual.mean_log_loss is not None
    assert global_player.mean_log_loss is not None


def test_two_humans_can_play_with_injected_input():
    answers_one = iter(["r"])
    answers_two = iter(["s"])
    player_one = HumanPlayer("Alice", input_function=lambda prompt: next(answers_one))
    player_two = HumanPlayer("Bob", input_function=lambda prompt: next(answers_two))

    records = play_match(player_one, player_two, rounds=1)

    assert records[0].player_one_payoff == 1

    summary = summarize_match(records)
    assert summary.player_one_wins == 1
    assert summary.player_two_wins == 0
    assert summary.draws == 0
    assert summary.player_one_mean_payoff == pytest.approx(1.0)
    assert summary.player_one_move_counts == (1, 0, 0)
    assert summary.player_two_move_counts == (0, 0, 1)
