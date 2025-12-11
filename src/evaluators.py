from src.decorators import evaluator, Evaluator
from jax import numpy as jnp
from collections.abc import MutableMapping
import jax
import pandas as pd
from typing import NamedTuple
from pathlib import Path
from itertools import product

import warnings


@evaluator(output="rmse", label="RMSE")
def rmse(true, target):
    return jnp.sqrt(jnp.mean((true - target) ** 2))


@evaluator(output="bias", label="Bias")
def bias(true, target):
    return jnp.mean(target - true)


@evaluator(output="mae", label="MAE")
def mae(true, target):
    return jnp.mean(jnp.abs(target - true))


def _get_stack_of_estimates(method_output: list[NamedTuple], target):
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
) -> pd.DataFrame:
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

            # this feels very heavy for what we are doing, since we probably aren't going to be jitting
            # or doing this over large numbers of methods
            evaluations = jax.vmap(evaluator_fn, in_axes=(None, 0))(
                target_vals, estimates
            )
            output_data = pd.DataFrame(
                {
                    "data_key": [data_key] * len(method_keys),
                    "method_key": method_keys,
                    "target": [target] * len(method_keys),
                    "evaluator": [evaluator_fn.label] * len(method_keys),
                    "evaluations": getattr(evaluations, evaluator_fn.output),
                }
            )
            eval_frames.append(output_data)

    full_results = pd.concat(eval_frames).pivot(
        index=["data_key", "method_key", "target"],
        columns="evaluator",
        values="evaluations",
    )

    if simulation_dir is not None:
        output_dir = Path(simulation_dir) / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        full_results.to_csv(output_dir / "evaluations.csv")

    return full_results
