import warnings
from collections.abc import MutableMapping
from itertools import product
from pathlib import Path
from typing import NamedTuple

import jax
import pandas as pd
from jax import numpy as jnp

from src.decorators import Evaluator, evaluator
from src.utils import get_params_from_scenario_keystring


@evaluator(output="rmse")
def rmse(true, target):
    """
    Standard RMSE evaluator.
    """
    return jnp.sqrt(jnp.mean((true - target) ** 2))


@evaluator(output="bias")
def bias(true, target):
    """
    Standard bias function.
    """
    return jnp.mean(target - true)


@evaluator(output="mae")
def mae(true, target):
    """
    Standard MAE function.
    """
    return jnp.mean(jnp.abs(target - true))


@evaluator(output="coverage", label="Coverage")
def coverage(true, target, sd, alpha=0.95):
    """
    Calculate coverage based on assumed normal confidence intervals.
    """
    low = target - sd * jax.scipy.stats.norm.ppf(1 - alpha / 2)
    high = target + sd * jax.scipy.stats.norm.ppf(1 - alpha / 2)
    return jnp.mean((true >= low) & (true <= high))


def _get_stack_of_estimates(method_output: list[NamedTuple], target):
    """
    Helper function to retrieve particular estimates for each method. For example,
    if each method outputs an object "beta_hat", then this creates a stack of beta estimates.

    For example, suppose we run 100 simulations for 5 coefficients across 3
    methods. This function will return a 3x100x5 tensor of beta_hat estimates.

    Args:
        method_output (list[NamedTuple]): The list of method output objects.
        target (str): The target variable name to extract (e.g., "beta").
    Returns:
        jnp.ndarray: A stacked array of estimates for the target variable.
    """
    method_output_fields = (
        method_output[0]._fields
        if isinstance(method_output, list)
        else method_output._fields
    )
    # this looks for things like "beta_hat" for "beta"; if the name of the
    # target is contained in the output field of a method, you're all set
    estimate_fields = [f for f in method_output_fields if target in f]
    if len(estimate_fields) > 1:
        raise ValueError(
            f"Multiple estimate fields found for target {target} in method output {method_output_fields}. Please use a single appropriate method output name, such as {target}_hat."
        )
    else:
        estimate_field = estimate_fields[0]

    estimate_stack = [getattr(estimates, estimate_field) for estimates in method_output]
    estimate_stack = jnp.stack(estimate_stack)
    return estimate_stack


def evaluate_methods(
    evaluators: Evaluator | list[Evaluator],
    data_dict: MutableMapping,
    method_dict: MutableMapping,
    targets: str | list[str],
    simulation_dir=None,
    prng_key=None,  # not used, but could be for things like bootstrap methods
) -> pd.DataFrame:
    """
    Evaluate fitted methods against true data using specified evaluators.
    Args:
        evaluators (Evaluator | list[Evaluator]): The evaluator function(s) to use.
        data_dict (MutableMapping): The dictionary of generated data objects.
        method_dict (MutableMapping): The dictionary of fitted method outputs.
        targets (str | list[str]): The target variable(s) to evaluate.
        simulation_dir (str | Path | None): Directory to save evaluation results.
        prng_key: Not used currently.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation results.

    Note: This function assumes that the output of any particular evaluator will be
    a single value (e.g., RMSE, bias). If an evaluator returns multiple values,
    this function may need to be adjusted accordingly.
    """

    if not isinstance(evaluators, (list, tuple)):
        evaluators = [evaluators]

    if isinstance(targets, str):
        targets = [targets]

    # get targets from the data
    eval_frames = []
    for data_key in data_dict.keys():
        # filter to only method runs associated with this dataset. If none are found,
        # raise a warning and continue
        relevant_methods = [
            (k, v) for k, v in method_dict.items() if f"{data_key}_" in k
        ]
        if not relevant_methods:
            warn_msg = f"No methods found for data key {data_key}. Skipping evaluations for this dataset. Consider re-running the estimation step."
            warnings.warn(warn_msg)
            continue
        method_keys, method_data = zip(*relevant_methods)

        # since we aren't looking at the function any more, get the
        # output fields directly from the outputs

        data_obj = data_dict[data_key]
        for target, evaluator_fn in product(targets, evaluators):
            target_vals = getattr(data_obj, target)
            estimates = _get_stack_of_estimates(list(method_data), target)

            # iterate over the estimates with a copy of the true values
            evaluations = jax.vmap(evaluator_fn, in_axes=(None, 0))(
                target_vals, estimates
            )
            output_data = pd.DataFrame(
                [
                    get_params_from_scenario_keystring(method_key)
                    for method_key in method_keys
                ]
                # TODO: this assumes the output of the evaluator is a single value.
                # If it is multiple, we need to expand this to assign each output
                # feature separately, with an appropriate name. This should mirror
                # the Method class pretty closely.
            ).assign(
                target=target,
                evaluator=evaluator_fn.label,
                value=getattr(evaluations, evaluator_fn.output),
            )
            eval_frames.append(output_data)

    eval_df = pd.concat(eval_frames)
    index_cols = set(eval_df.columns).difference({"evaluator", "value"})

    full_results = eval_df.pivot(
        index=index_cols,
        columns="evaluator",
        values="value",
    ).reset_index()

    if simulation_dir is not None:
        output_dir = Path(simulation_dir) / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        full_results.to_csv(output_dir / "evaluations.csv", index=False)

    return full_results
