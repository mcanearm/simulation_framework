import logging
from pathlib import Path
from typing import Union

import jax
from jaxtyping import PRNGKeyArray

from src.constants import VALID_KEY_NAMES
from src.decorators import DGP, Method
from src.utils import DiskDict, SimulationScenario, key_to_str

logger = logging.getLogger(__name__)

# What's needed to run with a DGP and Method?
# 1) output of the DGP saved to disk
# 2) Reuse the data with multiple parameters on the methods
# 3) Allocate a "chain" of steps that we can run sequentially
# Solution: DataCatalog and MethodCatalog classes that can save/load data and methods; Method Catalog takes a data catalog as input,
# but it should operate like a dictionar


def run_simulations(
    prng_key: PRNGKeyArray,
    scenarios: SimulationScenario,
    data_dir: Union[Path, str] = Path("./simulation/"),
    evaluations=None,
    plotters=None,
    n_sims: int = 100,
):
    """
    Run a fully vectorized simulation setup, given the DGP and the method. Note here that for each array of parameters,
    we need to vectorize over them and somehow work out the output dimensionality. I think we can use Xarray for this.

    You must pass in a valid PRNG key to begin the simulation process, and this is used to cache all data generation
    and model outputs appropriately.
    """

    data_gen_key, method_gen_key = jax.random.split(prng_key, 2)
    key_str = key_to_str(prng_key)

    # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
    # each other from run to run
    output_dir = Path(data_dir) / key_str / f"n_sims={n_sims}"

    data_store = DiskDict(output_dir / "data")
    method_store = DiskDict(output_dir / "methods")

    # iterate to start via generated data; once all data is generated, fit
    # each method on each dataset as it is generated.
    for data_scenario in scenarios.dgp:
        dgp_fn, dgp_params = data_scenario

        # overwrite the data_gen key on each iteration
        sim_gen_key, data_gen_key = jax.random.split(data_gen_key, 2)
        sim_gen_keys = jax.random.split(sim_gen_key, n_sims)
        bound = dgp_fn.sig.bind_partial(**dgp_params)
        bound.apply_defaults()
        dgp_batch_fn = jax.vmap(dgp_fn, in_axes=(0, *[None] * len(bound.arguments)))
        dgp_output = dgp_batch_fn(sim_gen_keys, *bound.arguments.values())

        data_store[data_scenario.simkey] = dgp_output

        for method_scenario in scenarios.method:
            # might not use these, but it's cheap and handles cases where randomization
            # is part of the model fit.
            method, method_kwargs = method_scenario
            method_fit_key, method_gen_key = jax.random.split(method_gen_key, 2)
            method_fit_keys = jax.random.split(method_fit_key, n_sims)

            if method._key_param_name:
                # since the prng key has to be first, this should bind to the first argument; but maybe bug?
                method_bind_params = {
                    **method_kwargs,
                    method._key_param_name: method_fit_keys,
                }
            else:
                method_bind_params = method_kwargs

            dgp_output_args = {
                k: v
                for k, v in zip(dgp_fn.output, dgp_output)
                if k in method.sig.parameters
            }
            bound = method.sig.bind_partial(**dgp_output_args, **method_bind_params)
            bound.apply_defaults()
            vmap_sig = tuple(
                0 if k in dgp_output_args or k == method._key_param_name else None
                for k in method.sig.parameters
            )

            method_batch_fn = jax.vmap(method, in_axes=vmap_sig)
            # batch over keys if necessary
            method_output = method_batch_fn(*bound.arguments.values())
            result_simkey = data_scenario.simkey + "_" + method_scenario.simkey
            method_store[result_simkey] = method_output

    return data_store, method_store
