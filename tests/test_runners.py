import pytest
from collections.abc import MutableMapping
from jax import numpy as jnp

from runners import run_simulations
from src.evaluators import rmse, bias
from tests.conftest import ols_data, ridge, ols


def test_sim_repeat(key, tmpdir):
    first_data_dir = tmpdir / "sim1"
    second_data_dir = tmpdir / "sim2"

    data_gen = [
        (ols_data, {"n": 50, "p": 5, "dist": ["normal", "t"]}),
    ]
    method_fit = [(ridge, {"alpha": [0.1, 1.0]}), (ols, {})]

    evaluators = [rmse, bias]

    simulated_outputs1 = run_simulations(
        key,
        data_gen,
        method_fit,
        evaluators,
        targets=["beta"],
        data_dir=first_data_dir,
    )
    simulated_outputs2 = run_simulations(
        key,
        data_gen,
        method_fit,
        evaluators,
        targets=["beta"],
        data_dir=second_data_dir,
    )

    for item1, item2 in zip(simulated_outputs1, simulated_outputs2):
        if isinstance(item1, MutableMapping):
            assert set(item1.keys()) == set(item2.keys())
            for key in item1.keys():
                for val1, val2 in zip(item1[key], item2[key]):
                    assert jnp.allclose(val1, val2)
        else:
            # if it isn't a mutable mapping, it's a pandas DF
            assert jnp.allclose(item1.values, item2.values)


@pytest.mark.skip(reason="Placeholder test")
def test_method_with_rng(key):
    pass
