import numpy as np
import pytest

from example.ridge_example import jax_ridge, linear_data_jax, linear_data_np, np_ridge
from src.dgp import generate_data
from src.methods import fit_methods


# TODO: it seems like a problem that I have to share n_sims across the entire chain; shouldn't
# that be decided based on the shape of the data_generating_process?
@pytest.fixture
def jax_datasets(key):
    return generate_data(
        key,
        [
            (linear_data_jax, {"p": [5, 10], "n": [50, 100]}),
        ],
        n_sims=100,
    )


@pytest.fixture
def np_datasets():
    rng = np.random.default_rng(0)
    return generate_data(
        rng,
        [
            (linear_data_np, {"p": [5, 10], "n": [50, 100]}),
        ],
        n_sims=100,
    )


@pytest.mark.parametrize(
    "dataset", ["jax_datasets", "np_datasets"], ids=["jax_data", "numpy_data"]
)
@pytest.mark.parametrize(
    "method",
    [jax_ridge, np_ridge],
    ids=["jax_method", "numpy_method"],
)
def test_fit_models(dataset, method, request):
    datadict = request.getfixturevalue(dataset)

    if dataset == "jax_datasets" and method == np_ridge:
        pytest.skip("Skipping incompatible numpy method with jax dataset")
    model_fits = fit_methods(
        model_mapping=[(method, {"alpha": [0.1, 1.0]})],
        data_dict=datadict,
    )
    assert isinstance(model_fits, dict)
