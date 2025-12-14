from src.runners import run_simulations
from tests.conftest import ols_data, ridge, ols
from src.evaluators import rmse, bias, mae
from src.plotters import create_plotter_fn
import seaborn as sns
from pathlib import Path
import pytest
import jax


@pytest.mark.parametrize("run_jit", [True, False], ids=["jit", "no_jit"])
def test_happy_path(key, tmpdir, run_jit):
    ols_data_fn = ols_data
    ridge_fn = ridge
    ols_fn = ols
    if run_jit:
        ols_data_fn = jax.jit(ols_data, static_argnames=["n", "p", "dist"])
        ridge_fn = jax.jit(ridge, static_argnames=["alpha"])
        ols_fn = jax.jit(ols)

    data, methods, evaluations, plots = run_simulations(
        key,
        dgp_mapping=[
            (
                ols_data_fn,
                {
                    "n": [100, 200],
                    "p": [5, 10, 20, 50],
                    "dist": ["normal", "t"],
                },
            )
        ],
        method_mapping=[
            (ridge_fn, {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}),
            (ols_fn, {}),
        ],
        evaluators=[rmse, bias, mae],
        plotters=[
            create_plotter_fn(
                x="p",
                y="rmse",
                hue="method",
                col="dist",
                plot_class=sns.lineplot,
            ),
            create_plotter_fn(
                x="method",
                y="mae",
                col="p",
                row="n",
                plot_class=sns.barplot,
            ),
        ],
        targets=["beta"],
        n_sims=100,
        simulation_dir=tmpdir / "example",
    )
    sim_dir = Path(tmpdir) / "example"

    len(list(sim_dir.glob("*/*/*")))
    # three diretories and the evaluation DF
    # lists starting key AND n_sims
    assert len(list(sim_dir.glob("*/*/*"))) == 4
