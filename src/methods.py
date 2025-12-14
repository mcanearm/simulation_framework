from jax._src.pjit import JitWrapped
import jax.typing
import jax
from pathlib import Path
from collections.abc import MutableMapping
from src.decorators import Method
from src.utils import (
    DiskDict,
    bind_arguments,
    create_vmap_signature,
    generate_scenarios,
)
from typing import Union
import tqdm
import numpy as np


def _execute(method_fn, method_input_args, method_kwargs, prng_key=None):
    exec_fn = None
    for v in method_input_args.values():
        if isinstance(v, np.ndarray):
            exec_fn = _np_execute
            break
        elif isinstance(v, jax.typing.ArrayLike):
            exec_fn = _execute_jax
            break
        else:
            continue
    if exec_fn is None:
        raise ValueError(
            "Could not determine execution backend from method input arguments."
        )
    return exec_fn(method_fn, method_input_args, method_kwargs, prng_key)


def _execute_jax(method_fn, method_input_args, method_kwargs, prng_key=None):
    if method_fn._requires_key and prng_key is not None:
        method_input_args[method_fn._key_param_name] = prng_key

    bound = bind_arguments(method_fn, **method_input_args, **method_kwargs)
    vmap_sig = create_vmap_signature(list(method_input_args.keys()), bound)
    method_batch_fn = jax.vmap(method_fn, in_axes=vmap_sig)
    method_output = method_batch_fn(*bound.arguments.values())
    return method_output


def _np_execute(method_fn, method_input_args, method_kwargs, rng=None):
    bound = bind_arguments(method_fn, **method_input_args, **method_kwargs)
    if rng is not None:
        bound.arguments[method_fn._key_param_name] = rng
    n_sims = next(iter(method_input_args.values())).shape[0]
    arg_calls = [
        {
            **{k: method_input_args[k][i, ...] for k in method_input_args.keys()},
            **method_kwargs,
        }
        for i in range(n_sims)
    ]
    method_results = [method_fn(**arg_call) for arg_call in arg_calls]

    stacked_outputs = tuple(
        [
            np.stack([getattr(item, output_name) for item in method_results], axis=0)
            for output_name in method_fn.output_class._fields
        ]
    )
    method_output = method_fn.output_class(*stacked_outputs)

    return method_output


def fit_methods(
    model_mapping: list[tuple[Method | JitWrapped, dict]],
    data_dict: MutableMapping,
    simulation_dir: Union[Path, str, None] = None,
    prng_key=None,
) -> MutableMapping[str, jax.typing.ArrayLike]:
    """
    Fit methods over generated data outputs.

    Args:
        model_mapping (list[tuple[Method, dict]]): List of tuples pairing Method functions
            with their parameter grids.
        data_dict (MutableMapping): The dictionary of generated data objects.
        simulation_dir (Union[Path, str, None]): Directory to save fitted method results.
        prng_key: JAX PRNG key for randomness in methods that require it.
    Returns:
        MutableMapping[str, jax.typing.ArrayLike]: A dictionary-like object containing fitted method outputs.
    """
    method_fit_key = prng_key

    # get the number of simulations from one of the data outputs; remember, they're
    # always tuples

    if simulation_dir is not None:
        # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
        # each other from run to run
        output_dir = Path(simulation_dir)
        method_store = DiskDict(output_dir / "methods")
    else:
        # use a plain dictionary if no data directory is provided
        method_store = dict()

    method_scenarios = [
        scenario
        for method_fn, param_set in model_mapping
        for scenario in generate_scenarios(method_fn, param_set)
    ]
    for data_key, dgp_output in tqdm.tqdm(
        data_dict.items(), position=0, unit="datasets"
    ):
        for method_scenario in tqdm.tqdm(
            method_scenarios, position=1, leave=False, unit="methods"
        ):
            result_simkey = data_key + "__" + method_scenario.simkey
            if result_simkey in method_store:
                continue

            method_fn, method_kwargs = method_scenario

            method_input_args = {
                arg: getattr(dgp_output, arg)
                for arg in dgp_output._fields
                if any([arg in in_arg for in_arg in method_fn.sig.parameters])
            }

            if method_fn._requires_key:
                try:
                    # split the provided key and overwrite it for the next method
                    # if one is required
                    method_fit_key, prng_key = jax.random.split(method_fit_key, 2)
                # TODO: modify to be real exception later
                # if attributeError, this is a numpy RNG style and we keep going
                except TypeError:
                    pass
                except Exception as e:
                    raise ValueError(
                        "A valid PRNG key must be provided to fit_models if any method requires a key."
                    ) from e

            method_output = _execute(
                method_fn, method_input_args, method_kwargs, method_fit_key
            )
            method_store[result_simkey] = method_output

    return method_store
