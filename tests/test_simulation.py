from bayesian_rps.opponents import UniformRandomOpponent
from bayesian_rps.simulation import simulate


def test_simulation_is_reproducible():
    first = simulate(UniformRandomOpponent(), rounds=20, seed=42)
    second = simulate(UniformRandomOpponent(), rounds=20, seed=42)
    assert first == second
