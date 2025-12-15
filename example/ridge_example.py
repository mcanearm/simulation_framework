import jax
import pandas as pd
import numpy as np
from jax import numpy as jnp

from src.decorators import method, dgp
from src.utils import function_timer
from src.evaluators import rmse, bias
from src.runners import run_simulations


@method(output="beta_hat", label="Ridge")
def jax_ridge(X, y, alpha=0.1):
    """
    test docstring
    """
    p = X.shape[1]
    beta_hat = jnp.linalg.inv(X.T @ X + alpha * jnp.eye(p)) @ X.T @ y
    return beta_hat


@method(output="beta_hat", label="OLS")
def jax_ols(X, y):
    """
    Ordinary Least Squares regression
    """
    beta_hat = jnp.linalg.inv(X.T @ X) @ X.T @ y
    return beta_hat


@dgp(output=["X", "y", "beta"], label="OLS")
def linear_data_jax(prng_key, n=100, p=10, noise=1.0, dist="normal"):
    """
    Generate linear regression data.
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    if dist == "normal":
        noise = jax.random.normal(prng_key, shape=(n,)) * noise
    elif dist == "t":
        noise = jax.random.t(prng_key, df=3, shape=(n,)) * noise
    noise = jax.random.normal(prng_key, shape=(n,)) * 0.5
    y = X @ true_beta + noise
    return X, y, true_beta


@dgp(output=["X", "y", "beta"], label="exponential_data")
def exponential_data_jax(prng_key, n=100, p=10, noise=1.0):
    """
    Generate data with an exponential relationship.
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    noise = jax.random.normal(prng_key, shape=(n,)) * noise
    y = jnp.exp(X @ true_beta / 10) + noise
    return X, y, true_beta


@dgp(output=["X", "y", "beta"], label="LinearNP")
def linear_data_np(rng, n=100, p=10, noise=1.0, dist="normal"):
    X = rng.normal(size=(n, p))
    true_beta = jnp.arange(1, p + 1)  # no randomization on beta
    if dist == "normal":
        noise = rng.normal(size=(n,)) * noise
    elif dist == "t":
        noise = rng.standard_t(df=3, size=(n,)) * noise
    y = X @ true_beta + noise
    return X, y, true_beta


@method(output="beta_hat", label="RidgeNP")
def np_ridge(X, y, alpha=0.1):
    """
    test docstring
    """
    p = X.shape[1]
    beta_hat = np.linalg.inv(X.T @ X + alpha * jnp.eye(p)) @ X.T @ y
    return beta_hat


if __name__ == "__main__":
    seed = 20250607
    k1, k2 = jax.random.split(jax.random.key(20250607), 2)
    np_rng = np.random.default_rng(20250607)

    def make_config(dgp, method, label):
        return {
            "dgp_mapping": [
                (
                    dgp,
                    {
                        "n": [100, 200],
                        "p": [10, 20],
                        "dist": ["normal", "t"],
                    },
                ),
            ],
            "method_mapping": [
                (method, {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}),
            ],
            "evaluators": [rmse, bias],
            "targets": ["beta"],
            "label": label,
            "allow_cache": False,
            "seed": seed,  # add a seed label
        }

    for (key, dgp_fn, method_fn), label in zip(
        [
            (np_rng, linear_data_np, np_ridge),
            (k1, linear_data_jax, jax_ridge),
            (
                k2,
                jax.jit(linear_data_jax, static_argnames=("n", "p", "noise", "dist")),
                jax.jit(jax_ridge),
            ),
        ],
        ["np_ridge", "jax_ridge", "jax_ridge_jit"],
    ):
        timings = []
        device = jax.devices()[0].device_kind
        for n_sims in [10, 50, 100, 500, 1000, 2000, 5000]:
            with function_timer() as sim_timer:
                _, _, _, plots = run_simulations(
                    key,
                    **make_config(dgp_fn, method_fn, label),
                    simulation_dir="./example/_simulations/",
                    n_sims=n_sims,
                )
            print(
                f"{label} simulations (N_sims={n_sims}) took {sim_timer.elapsed_time:.2f} seconds."
            )
            timings.append(
                {
                    "label": label,
                    "device": device,
                    "n_sims": n_sims,
                    "elapsed_time": sim_timer.elapsed_time,
                }
            )
    timings_df = pd.DataFrame(timings)
    timings_df.to_csv("example/timings.csv", index=False)
