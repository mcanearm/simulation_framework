import logging
from pathlib import Path
from typing import Union

import jax
from jaxtyping import PRNGKeyArray, Array

from decorators import DGP, Evaluator, Method
from src.utils import DiskDict, key_to_str
from collections.abc import MutableMapping

from utils import bind_arguments, create_vmap_signature

logger = logging.getLogger(__name__)


def run_simulations(
    prng_key: PRNGKeyArray,
    dgp_mapping: list[tuple[DGP, dict]],
    method_mapping: list[tuple[Method, dict]],
    evaluator_mapping: list[tuple[Evaluator, dict]],
    n_sims: int = 100,
    data_dir: Union[Path, str, None] = None,
) -> tuple[MutableMapping[str, Array], MutableMapping[str, Array]]:
    """
    Run a fully vectorized simulation setup, given the DGP and the method. Note here that for each array of parameters,
    we need to vectorize over them and somehow work out the output dimensionality. I think we can use Xarray for this.

    You must pass in a valid PRNG key to begin the simulation process, and this is used to cache all data generation
    and model outputs appropriately.
    """

    data_gen_key, method_gen_key = jax.random.split(prng_key, 2)
    key_str = key_to_str(prng_key)

    if data_dir is not None:
        # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
        # each other from run to run
        output_dir = Path(data_dir) / key_str / f"n_sims={n_sims}"
        data_store = DiskDict(output_dir / "data")
        method_store = DiskDict(output_dir / "methods")
    else:
        # use a plain dictionary if not data directory is provided
        data_store = dict()
        method_store = dict()

    # iterate to start via generated data; once all data is generated, fit
    # each method on each dataset as it is generated.
    for data_scenario in scenarios.dgp:
        dgp_fn, dgp_params = data_scenario

        # overwrite the data_gen key on each iteration so that the data
        # is different on the next split.
        sim_gen_key, data_gen_key = jax.random.split(data_gen_key, 2)
        sim_gen_keys = jax.random.split(sim_gen_key, n_sims)
        dgp_argset = {dgp_fn._key_param_name: sim_gen_keys, **dgp_params}

        bound = bind_arguments(dgp_fn, **dgp_argset)
        dgp_vmap = create_vmap_signature(dgp_fn._key_param_name, bound)
        dgp_batch_fn = jax.vmap(
            dgp_fn,
            in_axes=dgp_vmap,
        )
        dgp_output = dgp_batch_fn(*bound.arguments.values())
        dgp_output_args = {k: v for k, v in zip(dgp_fn.output, dgp_output)}

        data_store[data_scenario.simkey] = dgp_output

        # NOTE: It may be marginally preferable to move this outside the data loop
        # so that ALL data is generated, and then the methods are fit. However, since
        # the keys are deterministic and we use a different stream for both methods and
        # data, this should be fine.
        for method_scenario in scenarios.method:
            method_input_args = (
                [dgp_fn.output] if isinstance(dgp_fn.output, str) else dgp_fn.output
            )
            method, method_kwargs = method_scenario
            method_bind_params = {
                **method_kwargs,
                **dgp_output_args,
            }

            if method._requires_key:
                method_fit_key, method_gen_key = jax.random.split(method_gen_key, 2)
                method_fit_keys = jax.random.split(method_fit_key, n_sims)
                method_bind_params[method._key_param_name] = method_fit_keys
                method_input_args.append(method._key_param_name)

            bound = bind_arguments(method, **method_bind_params)
            vmap_sig = create_vmap_signature(method_input_args, bound)
            method_batch_fn = jax.vmap(method, in_axes=vmap_sig)
            method_output = method_batch_fn(*bound.arguments.values())
            result_simkey = data_scenario.simkey + "__" + method_scenario.simkey
            method_store[result_simkey] = method_output

    #
    return data_store, method_store
