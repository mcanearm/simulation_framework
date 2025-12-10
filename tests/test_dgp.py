from src.dgp import generate_data
from tests.conftest import ols_data
from collections.abc import MutableMapping


def test_data_generation(key):
    output = generate_data(
        key, [(ols_data, {"n": [5, 10], "N": [50, 100]})], n_sims=1000
    )

    assert isinstance(output, MutableMapping)
