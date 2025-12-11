from src.runners import run_simulations
from tests.conftest import ols_data, ridge, ols
from src.evaluators import rmse, bias, mae
import jax


# ols_data = jax.jit(ols_data, static_argnames=["n", "p", "dist"])
ridge = jax.jit(ridge)
ols = jax.jit(ols)

key = jax.random.PRNGKey(19900330)

sim_output = run_simulations(
    key,
    dgp_mapping=[
        (
            ols_data,
            {"n": [100, 200, 500], "p": [5, 10, 20, 50, 100], "dist": ["normal", "t"]},
        )
    ],
    method_mapping=[(ridge, {"alpha": [0.1, 0.5, 1.0, 10.0]}), (ols, {})],
    evaluators=[rmse, bias, mae],
    targets=["beta"],
    n_sims=5000,
    simulation_dir="example/ridge_vs_ols_results",
)
