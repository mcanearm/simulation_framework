import numpy as np
from src.runners import SimRunner
from src.decorators import data_generator, model

rng = np.random.default_rng(20250607)


@data_generator("Linear Data", output=("X", "y", "beta"))
def ols_data(N, p):
    X = rng.normal(size=(N, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(size=N)
    return X, y, beta


@model("Ridge", output="beta")
def ridge_model(X, y, lam=1.0):
    p = X.shape[1]
    eye = np.eye(p)

    beta = np.linalg.solve(X.T @ X + lam * eye, X.T @ y)
    return beta


def test_runner():
    # runner = SimRunner(ols_data, ridge_model, {"Ridge": {"lam": 0.1}})
    N_sims = 10

    Runner = SimRunner(
        data_generator=ols_data,
        method=ridge_model,
        sim_params={
            "Linear Data": {"N": [50, 100], "p": [5, 10, 20]},
            "Ridge": {"lam": [0.1, 0.5, 1.0, 1.5]},
        },
    )

    results = Runner.run_simulations(N_sims)
    assert len(results) == 2 * 3 * 4
    # TODO: come up with better tests here to make sure I have the right
    # scenarios
