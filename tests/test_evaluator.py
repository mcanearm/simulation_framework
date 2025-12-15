import jax
import pandas as pd
import pytest

from example.ridge_example import jax_ols, jax_ridge, linear_data_jax
from src.dgp import generate_data
from src.evaluators import bias, evaluate_methods, mae, rmse
from src.methods import fit_methods


@pytest.fixture(scope="module")
def simulation_set():
    key = jax.random.PRNGKey(0)

    generated_data = generate_data(
        key, [(linear_data_jax, {"n": 100, "p": 10})], n_sims=100
    )
    fitted_methods = fit_methods(
        [(jax_ridge, {"alpha": [0.1, 1.0, 10.0]}), (jax_ols, {})],
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
