import pytest
from tests.conftest import ridge, ols_data, np_ols, np_ridge
from src.dgp import generate_data
from src.methods import fit_methods
import numpy as np


# TODO: it seems like a problem that I have to share n_sims across the entire chain; shouldn't
# that be decided based on the shape of the data_generating_process?
@pytest.fixture
def jax_datasets(key):
    return generate_data(
        key,
        [
            (ols_data, {"p": [5, 10], "n": [50, 100]}),
        ],
        n_sims=100,
    )


@pytest.fixture
def np_datasets():
    rng = np.random.default_rng(0)
    return generate_data(
        rng,
        [
            (np_ols, {"p": [5, 10], "n": [50, 100]}),
        ],
        n_sims=100,
    )


@pytest.mark.parametrize(
    "dataset,method",
    [("jax_datasets", ridge), ("np_datasets", np_ridge)],
    ids=["jax", "numpy"],
)
def test_fit_models(dataset, method, request):
    datadict = request.getfixturevalue(dataset)
    model_fits = fit_methods(
        model_mapping=[(method, {"alpha": [0.1, 1.0]})],
        data_dict=datadict,
    )
    assert isinstance(model_fits, dict)
