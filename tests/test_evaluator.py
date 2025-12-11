from tests.conftest import ols_data, ols, ridge
from src.evaluators import rmse, evaluate_methods, mae, bias
import pytest
from src.methods import fit_methods
from src.dgp import generate_data
import pandas as pd
import jax


@pytest.fixture(scope="module")
def simulation_set():
    key = jax.random.PRNGKey(0)

    generated_data = generate_data(key, [(ols_data, {"n": 100, "p": 10})], n_sims=100)
    fitted_methods = fit_methods(
        [(ridge, {"alpha": [0.1, 1.0, 10.0]}), (ols, {})],
        data_dict=generated_data,
    )

    return generated_data, fitted_methods


@pytest.mark.parametrize(
    "evaluators",
    [rmse, [mae], [rmse, mae, bias]],
    ids=["single", "list_single", "list_multiple"],
)
def test_evaluators(simulation_set, evaluators):
    sim_data, method_fits = simulation_set
    evaluations = evaluate_methods(
        evaluators=evaluators,
        data_dict=sim_data,
        method_dict=method_fits,
        targets="beta",
    )
    assert isinstance(evaluations, pd.DataFrame)
    for col in ["method", "data"]:
        assert col in evaluations.columns

    if not isinstance(evaluators, list):
        evaluators = [evaluators]
    labels = [evaluator.label for evaluator in evaluators]

    assert all(label in evaluations.columns for label in labels)
