import logging
from pathlib import Path
from typing import Union
import pandas as pd

import jax
from jaxtyping import PRNGKeyArray, Array

from src.decorators import DGP, Evaluator, Method
from src.utils import key_to_str
from src.dgp import generate_data
from src.methods import fit_methods
from src.evaluators import evaluate_methods
from collections.abc import MutableMapping


logger = logging.getLogger(__name__)


def run_simulations(
    prng_key: PRNGKeyArray,
    dgp_mapping: list[tuple[DGP, dict]],
    method_mapping: list[tuple[Method, dict]],
    evaluators: list[Evaluator],
    targets: list[str],
    plotters: object | list[object] | None = None,
    n_sims: int = 100,
    simulation_dir: Union[Path, str, None] = None,
) -> tuple[MutableMapping[str, Array], MutableMapping[str, Array], pd.DataFrame, list]:
    """
    Run a fully vectorized simulation setup, given the DGP and the method. Note here that for each array of parameters,
    we need to vectorize over them and somehow work out the output dimensionality. I think we can use Xarray for this.

    You must pass in a valid PRNG key to begin the simulation process, and this is used to cache all data generation
    and model outputs appropriately.
    """

    data_gen_key, method_gen_key, evaluator_key = jax.random.split(prng_key, 3)
    key_str = key_to_str(prng_key)

    if simulation_dir is not None:
        # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
        # each other from run to run
        output_dir = Path(simulation_dir) / key_str / f"n_sims={n_sims}"
    else:
        # use a plain dictionary if no data directory is provided
        output_dir = None

    # iterate to start via generated data; once all data is generated, fit
    # each method on each dataset as it is generated.

    data_set = generate_data(
        data_gen_key, dgp_mapping, n_sims=n_sims, simulation_dir=output_dir
    )
    fitted_methods = fit_methods(
        method_mapping,
        data_dict=data_set,
        simulation_dir=output_dir,
        prng_key=method_gen_key,
    )
    evaluations = evaluate_methods(
        evaluators,
        data_set,
        fitted_methods,
        targets=targets,
        simulation_dir=output_dir,
        prng_key=evaluator_key,
    )

    plots = []
    if plotters:
        if not isinstance(plotters, list):
            plotters = [plotters]
        plots = [
            plotter(evaluations, simulation_dir=output_dir)  # type: ignore
            for plotter in plotters
        ]
    return data_set, fitted_methods, evaluations, plots
