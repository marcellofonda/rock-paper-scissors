from bayesian_rps.opponents import RandomOpponent
from bayesian_rps.simulation import simulate


def test_simulation_is_reproducible():
    first = simulate(RandomOpponent(), rounds=20, seed=42)
    second = simulate(RandomOpponent(), rounds=20, seed=42)
    assert first == second

