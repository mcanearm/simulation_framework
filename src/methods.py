import jax.typing
import jax
from pathlib import Path
from collections.abc import MutableMapping
from src.decorators import Method
from src.utils import (
    DiskDict,
    Scenario,
    get_arg_combinations,
    bind_arguments,
    create_vmap_signature,
)
from typing import Union


def fit_methods(
    model_mapping: list[tuple[Method, dict]],
    data_dict,
    simulation_dir: Union[Path, str, None] = None,
    prng_key=None,
) -> MutableMapping[str, jax.typing.ArrayLike]:
    """
    Run a fully vectorized simulation setup, given the DGP and the method. Note here that for each array of parameters,
    we need to vectorize over them and somehow work out the output dimensionality. I think we can use Xarray for this.

    You must pass in a valid PRNG key to begin the simulation process, and this is used to cache all data generation
    and model outputs appropriately.
    """

    # get the number of simulations from one of the data outputs; remember, they're
    # always tuples
    n_sims = next(iter(data_dict.values()))[0].shape[0]

    if simulation_dir is not None:
        # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
        # each other from run to run
        output_dir = Path(simulation_dir) / f"n_sims={n_sims}"
        method_store = DiskDict(output_dir / "methods")
    else:
        # use a plain dictionary if no data directory is provided
        method_store = dict()

    # iterate to start via generated data; once all data is generated, fit
    # each method on each dataset as it is generated.
    method_scenarios = [
        Scenario(method_fn, method_kwargs)
        for method_fn, param_set in model_mapping
        for method_kwargs in get_arg_combinations(param_set)
    ]
    for data_key, dgp_output in data_dict.items():
        for method_scenario in method_scenarios:
            method_fn, method_kwargs = method_scenario

            method_input_args = {
                arg: getattr(dgp_output, arg)
                for arg in dgp_output._fields
                if any([arg in in_arg for in_arg in method_fn.sig.parameters])
            }

            if method_fn._requires_key:
                try:
                    method_fit_key, prng_key = jax.random.split(prng_key, 2)
                # TODO: modify to be real exception later
                except Exception as e:
                    raise ValueError(
                        "A valid PRNG key must be provided to fit_models if any method requires a key."
                    ) from e
                method_fit_keys = jax.random.split(method_fit_key, n_sims)
                method_input_args[method_fn._key_param_name] = method_fit_keys

            method_bind_params = {**method_kwargs, **method_input_args}
            bound = bind_arguments(method_fn, **method_bind_params)
            vmap_sig = create_vmap_signature(list(method_input_args.keys()), bound)
            method_batch_fn = jax.vmap(method_fn, in_axes=vmap_sig)
            method_output = method_batch_fn(*bound.arguments.values())
            result_simkey = data_key + "__" + method_scenario.simkey
            method_store[result_simkey] = method_output

    return method_store
