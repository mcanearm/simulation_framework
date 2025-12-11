import pytest
from example.greenland import (
    generate_design_matrix,
    fit_mle,
)
from src.dgp import generate_data
from src.evaluators import evaluate_methods, rmse, bias, mae
from src.methods import fit_methods
import jax


@pytest.mark.skip(reason="Greenland paper hard to implement quickly")
def test_run_simulation(key, tmpdir):
    data_outputs = generate_data(
        key,
        [
            (
                jax.jit(
                    generate_design_matrix,
                    static_argnames=["N", "n", "rho", "sigma2", "tau0", "tau1"],
                ),
                {"N": [100, 500], "n": [10, 50]},
            )
        ],
        n_sims=8000,
        simulation_dir=tmpdir,
    )

    fitted_methods = fit_methods(
        data_dict=data_outputs,
        model_mapping=[(jax.jit(fit_mle, static_argnames=["max_iter", "ridge"]), {})],
        simulation_dir=tmpdir,
    )

    evaluations = evaluate_methods(
        data_dict=data_outputs,
        method_dict=fitted_methods,
        targets="beta",
        evaluators=[rmse, bias, mae],
    )

    assert evaluations is not None
