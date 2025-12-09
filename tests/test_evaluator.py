from conftest import ols_data, ols, ridge
from src.evaluators import rmse, evaluate_methods
from src.runners import fit_models
from src.utils import simulation_grid
import pandas as pd


def test_rmse(key):
    scenarios = simulation_grid(
        dgps=[(ols_data, {"n": 100, "p": 10})],
        methods=[(ridge, {"alpha": [0.1, 1.0, 10.0]}), (ols, {})],
    )

    sim_data, method_fits = fit_models(
        key,
        scenarios,
        n_sims=100,
    )

    evaluations = evaluate_methods(
        evaluators=rmse, data_dict=sim_data, method_dict=method_fits, targets="beta"
    )
    assert isinstance(evaluations, pd.DataFrame)
    for col in ["data_key", "method_key", "target"]:
        assert col in evaluations.index.names
