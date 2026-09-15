import pytest

from bayesian_rps.game import Move
from bayesian_rps.model import ContextualDirichletModel


def test_prior_predictive_is_uniform():
    model = ContextualDirichletModel()
    prediction = model.predict((Move.ROCK, Move.PAPER))
    assert prediction == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_update_only_changes_the_relevant_context():
    model = ContextualDirichletModel()
    context = (Move.ROCK, Move.PAPER)
    model.update(context, Move.SCISSORS)

    assert model.predict(context) == pytest.approx((0.25, 0.25, 0.50))
    assert model.predict((Move.PAPER, Move.ROCK)) == pytest.approx((1 / 3,) * 3)

