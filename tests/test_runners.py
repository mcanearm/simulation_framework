import pytest
from jax import numpy as jnp

from src.runners import run_simulations
from src.utils import simulation_grid
from tests.conftest import exponential_data, ols_data, ridge


def test_run_simulations(key, tmpdir):
    scenarios = simulation_grid(
        dgps=(ols_data, {"n": [50, 100], "p": [5, 10], "dist": ["normal", "t"]}),
        methods=[(ridge, {"alpha": [0.1, 1.0]})],
    )
    data_output, method_output = run_simulations(
        key, scenarios, data_dir=tmpdir, n_sims=100
    )

    assert len(list(data_output.keys())) == 8
    assert len(list(method_output.keys())) == 16  # ran two models for each gp process


def test_sim_repeat(key, tmpdir):
    scenarios = simulation_grid(
        dgps=(exponential_data, {"n": [50], "p": [5]}),
        methods=[(ridge, {"alpha": [0.5]})],
    )

    first_data_dir = tmpdir / "sim1"
    second_data_dir = tmpdir / "sim2"

    data_output1, method_output1 = run_simulations(
        key, scenarios, data_dir=first_data_dir, n_sims=50
    )
    data_output2, method_output2 = run_simulations(
        key, scenarios, data_dir=second_data_dir, n_sims=50
    )

    # Check that outputs are the same
    for k in data_output1.keys():
        data1 = data_output1[k]
        data2 = data_output2[k]
        for arr1, arr2 in zip(data1, data2):
            assert jnp.array_equal(arr1, arr2)

    for k in method_output1.keys():
        method1 = method_output1[k]
        method2 = method_output2[k]
        assert jnp.array_equal(method1, method2)


@pytest.mark.skip(reason="Placeholder test")
def test_method_with_rng(key):
    pass
