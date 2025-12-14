import logging
from pathlib import Path
from typing import Union
from jax._src.pjit import JitWrapped
import pandas as pd

import jax
from jaxtyping import PRNGKeyArray, ArrayLike

from src.decorators import DGP, Evaluator, Method
from src.utils import key_to_str
from src.dgp import generate_data
from src.methods import fit_methods
from src.evaluators import evaluate_methods
from collections.abc import MutableMapping


logger = logging.getLogger(__name__)


def run_simulations(
    prng_key: PRNGKeyArray,
    dgp_mapping: list[tuple[DGP | JitWrapped, dict]],
    method_mapping: list[tuple[Method | JitWrapped, dict]],
    evaluators: list[Evaluator],
    targets: list[str],
    plotters: object | list[object] | None = None,
    n_sims: int = 100,
    simulation_dir: Union[Path, str, None] = None,
) -> tuple[
    MutableMapping[str, ArrayLike], MutableMapping[str, ArrayLike], pd.DataFrame, list
]:
    """
    Run a simulation setup for all combinations of data generating processes and
    methods, evaluate the methods, and optionally create plots. If a simulation directory
    is provided, intermediate and final results are saved to disk.

    Args:
        prng_key (PRNGKeyArray): JAX PRNG key for randomness.
        dgp_mapping (list[tuple[DGP | JitWrapped, dict]]): List of tuples pairing DGP functions
            with their parameter grids.
        method_mapping (list[tuple[Method | JitWrapped, dict]]): List of tuples pairing Method functions
            with their parameter grids.
        evaluators (list[Evaluator]): List of evaluator functions to assess method performance.
        targets (list[str]): List of target variable names to evaluate.
        plotters (object | list[object] | None): Plotter function(s) to create visualizations.
        n_sims (int): Number of simulations to run for each scenario.
        simulation_dir (Union[Path, str, None]): Directory to save simulation results.
    Returns:
        tuple: A tuple containing:
            - data_set (MutableMapping[str, Array]): Generated data sets.
            - fitted_methods (MutableMapping[str, Array]): Fitted method outputs.
            - evaluations (pd.DataFrame): DataFrame of evaluation results.
            - plots (list): List of generated plots for further modification.
    """

    key_str = key_to_str(prng_key)

    try:
        data_gen_key, method_gen_key, evaluator_key = jax.random.split(prng_key, 3)
    except TypeError:
        # if type error, it's a numpy rng
        data_gen_key = method_gen_key = evaluator_key = prng_key

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
