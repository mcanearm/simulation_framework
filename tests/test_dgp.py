import pytest
from src.dgp import generate_data
from tests.conftest import ols_data, np_ols
from collections.abc import MutableMapping
import jax
import numpy as np


jax_key = jax.random.PRNGKey(0)
np_rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    "dgp_fn, rng_key",
    [(ols_data, jax_key), (np_ols, np_rng)],
    ids=["jax_dgp", "numpy_dgp"],
)
def test_data_generation(rng_key, dgp_fn):
    output = generate_data(
        rng_key, [(dgp_fn, {"n": [5, 10], "N": [50, 100]})], n_sims=1000
    )

    assert isinstance(output, MutableMapping)
    for key, data in output.items():
        assert data[0].shape[0] == 1000
