from conftest import ols_data, ols, ridge
from src.evaluators import rmse, evaluate_methods, mae, bias
import pytest
from src.runners import fit_models
from src.utils import simulation_grid
import pandas as pd
import jax


@pytest.fixture(scope="module")
def simulation_set():
    key = jax.random.PRNGKey(0)
    scenarios = simulation_grid(
        dgps=[(ols_data, {"n": 100, "p": 10})],
        methods=[(ridge, {"alpha": [0.1, 1.0, 10.0]}), (ols, {})],
    )

    sim_data, method_fits = fit_models(
        key,
        scenarios,
        n_sims=100,
    )
    return sim_data, method_fits


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
    for col in ["data_key", "method_key", "target"]:
        assert col in evaluations.index.names

    if not isinstance(evaluators, list):
        evaluators = [evaluators]
    labels = [evaluator.label for evaluator in evaluators]

    assert all(label in evaluations.columns for label in labels)
