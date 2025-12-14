from src.runners import run_simulations
from tests.conftest import ols_data, ridge, np_ridge, np_ols
from src.evaluators import rmse, bias, mae
from src.plotters import create_plotter_fn
import seaborn as sns
from pathlib import Path
import pytest
import jax
import numpy as np

my_key = jax.random.PRNGKey(0)
rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    "key_dgp_method_jit",
    [
        [rng, np_ols, np_ridge, False],
        [my_key, ols_data, ridge, False],
        [my_key, ols_data, ridge, True],
    ],
    ids=["numpy", "jax-no-jit", "jax-jit"],
)
def test_happy_path(tmpdir, key_dgp_method_jit):
    key, data_fn, method_fn, run_jit = key_dgp_method_jit
    if run_jit:
        data_fn = jax.jit(data_fn, static_argnames=["n", "p", "dist"])
        method_fn = jax.jit(method_fn, static_argnames=["alpha"])

    data, methods, evaluations, plots = run_simulations(
        key,
        dgp_mapping=[
            (
                data_fn,
                {
                    "n": [100, 200],
                    "p": [5, 10, 20, 50],
                    "dist": ["normal", "t"],
                },
            )
        ],
        method_mapping=[
            (method_fn, {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}),
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
