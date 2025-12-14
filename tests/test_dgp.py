import pytest
from src.dgp import generate_data
from example.ridge_example import linear_data_jax, linear_data_np
from collections.abc import MutableMapping
import jax
import numpy as np


jax_key = jax.random.key(0)
np_rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    "dgp_fn, rng_key",
    [(linear_data_jax, jax_key), (linear_data_np, np_rng)],
    ids=["jax_dgp", "numpy_dgp"],
)
def test_data_generation(rng_key, dgp_fn):
    output = generate_data(
        rng_key, [(dgp_fn, {"n": [5, 10], "N": [50, 100]})], n_sims=1000
    )

    assert isinstance(output, MutableMapping)
    for key, data in output.items():
        assert data[0].shape[0] == 1000
