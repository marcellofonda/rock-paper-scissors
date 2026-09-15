import random

import pytest

from bayesian_rps.game import Move
from bayesian_rps.policy import choose_move, expected_utilities


def test_expected_utilities():
    utilities = expected_utilities((0.60, 0.25, 0.15))
    assert utilities == pytest.approx((-0.10, 0.45, -0.35))
    assert choose_move((0.60, 0.25, 0.15), random.Random(0)) == Move.PAPER

