from src.decorators import evaluator, Evaluator
from jax import numpy as jnp
from collections.abc import MutableMapping
from src.utils import get_scenario_params
import jax
import pandas as pd
from typing import NamedTuple
from itertools import product

import warnings


@evaluator(output="rmse", label="RMSE")
def rmse(true, target):
    return jnp.sqrt(jnp.mean((true - target) ** 2))


def format_evaluator_to_df(data_key, method_key, target, evaluator, evaluations):
    """
    Lots of parameters, but actually just a helper function to everything into a long dataframe
    """
    nrows = len(getattr(evaluations, evaluator.output))

    method_label, method_params = get_scenario_params(method_key)
    data_label, method_params = get_scenario_params(data_key)

    return pd.DataFrame(
        {
            "data_key": [data_key] * len(method_key),
            "method_key": method_key,
            "target": [target] * len(method_key),
            "evaluator": [evaluator.label] * len(method_key),
            "evaluations": getattr(evaluations, evaluator.output),
        }
    )


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

    estimate_stack = jnp.stack(
        [getattr(estimates, estimate_field) for estimates in method_output]
    )
    return estimate_stack


def evaluate_methods(
    evaluators: Evaluator | list[Evaluator],
    data_dict: MutableMapping,
    method_dict: MutableMapping,
    targets: str | list[str],
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
        relevant_methods = [(k, v) for k, v in method_dict.items() if data_key in k]
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

    return full_results
