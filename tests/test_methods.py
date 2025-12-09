import pytest
from conftest import ridge, ols_data
from src.dgp import generate_data
from src.methods import fit_methods


# TODO: it seems like a problem that I have to share n_sims across the entire chian; shouldn't
# that be decided based on the shape of the data_generating_process?
@pytest.fixture
def datasets(key):
    return generate_data(key, [(ols_data, {"n": [5, 10], "N": [50, 100]})], n_sims=100)


def test_fit_models(datasets):
    model_fits = fit_methods(
        model_mapping=[(ridge, {"alpha": [0.1, 1.0]})],
        data_dict=datasets,
    )
    assert isinstance(model_fits, dict)
