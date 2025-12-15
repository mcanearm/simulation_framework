from collections.abc import MutableMapping

import pandas as pd
import pytest
from jax import numpy as jnp

from example.ridge_example import jax_ols, jax_ridge, linear_data_jax
from src.evaluators import bias, rmse
from src.runners import run_simulations


def test_sim_repeat(key, tmpdir):
    first_simulation_dir = tmpdir / "sim1"
    second_simulation_dir = tmpdir / "sim2"

    data_gen = [
        (linear_data_jax, {"n": 50, "p": 5, "dist": ["normal", "t"]}),
    ]
    method_fit = [(jax_ridge, {"alpha": [0.1, 1.0]}), (jax_ols, {})]

    evaluators = [rmse, bias]

    simulated_outputs1 = run_simulations(
        key,
        data_gen,
        method_fit,
        evaluators,
        targets=["beta"],
        simulation_dir=first_simulation_dir,
    )
    simulated_outputs2 = run_simulations(
        key,
        data_gen,
        method_fit,
        evaluators,
        targets=["beta"],
        simulation_dir=second_simulation_dir,
    )

    for item1, item2 in zip(simulated_outputs1, simulated_outputs2):
        if isinstance(item1, MutableMapping):
            assert set(item1.keys()) == set(item2.keys())
            for key in item1.keys():
                for val1, val2 in zip(item1[key], item2[key]):
                    assert jnp.allclose(val1, val2)
        elif isinstance(item1, pd.DataFrame):
            # if it isn't a mutable mapping, it's a pandas DF
            pd.DataFrame.equals(item1, item2)
        else:
            pass  # skip the plots list


@pytest.mark.skip(reason="Placeholder test")
def test_method_with_rng(key):
    pass
