from src.runners import run_simulations
from tests.conftest import ols_data, ridge, ols
from src.evaluators import rmse, bias, mae
from src.plotters import create_plotter_fn
import jax
import seaborn as sns


# ols_data = jax.jit(ols_data, static_argnames=["n", "p", "dist"])
# ridge = jax.jit(ridge)
# ols = jax.jit(ols)

key = jax.random.PRNGKey(19900330)


data, methods, evaluations, plots = run_simulations(
    key,
    dgp_mapping=[
        (
            ols_data,
            {"n": [100, 200, 500, 2000], "p": [5, 10, 20, 50], "dist": ["normal", "t"]},
        )
    ],
    method_mapping=[(ridge, {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}), (ols, {})],
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
            col="n",
            row="p",
            plot_class=sns.barplot,
        ),
    ],
    targets=["beta"],
    n_sims=100,
    simulation_dir="example/_ridge_vs_ols_results",
)
# evaluations = pd.read_csv(
#     "example/_ridge_vs_ols_results/key=0-19900330/n_sims=8000/evaluations/evaluations.csv"
# )
