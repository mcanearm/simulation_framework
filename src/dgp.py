from src.decorators import DGP
from src.utils import (
    get_arg_combinations,
    DiskDict,
    Scenario,
    bind_arguments,
    create_vmap_signature,
)
from pathlib import Path
import jax
import tqdm
import logging

logger = logging.getLogger(__name__)


def generate_data(
    prng_key,
    dgp_param_map: list[tuple[DGP, dict]],
    n_sims: int,
    simulation_dir=None,
):
    data_gen_key, _ = jax.random.split(prng_key, 2)

    if simulation_dir is not None:
        # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
        # each other from run to run
        output_dir = Path(simulation_dir)
        data_store = DiskDict(output_dir / "data")  # creates dir if it doesn't exist
    else:
        # use a plain dictionary if no data directory is provided
        data_store = dict()

    # iterate to start via generated data; once all data is generated, fit
    # each method on each dataset as it is generated.
    scenarios = [
        Scenario(dgp_fn, dgp_args)
        for dgp_fn, param_set in dgp_param_map
        for dgp_args in get_arg_combinations(param_set)
    ]
    logger.info(f"{len(scenarios)} scenarios generated.")
    for scenario in tqdm.tqdm(scenarios, unit="datasets"):
        dgp_fn, dgp_args = scenario
        if scenario.simkey in data_store:
            logger.info(
                f"Data for scenario {scenario.simkey} already exists; skipping generation."
            )
            continue
        # overwrite the data_gen key on each iteration so that the data
        # is different on the next split.
        sim_gen_key, data_gen_key = jax.random.split(data_gen_key, 2)
        sim_gen_keys = jax.random.split(sim_gen_key, n_sims)
        dgp_argset = {dgp_fn._key_param_name: sim_gen_keys, **dgp_args}

        bound = bind_arguments(dgp_fn, **dgp_argset)
        dgp_vmap = create_vmap_signature(dgp_fn._key_param_name, bound)
        dgp_batch_fn = jax.vmap(
            dgp_fn,
            in_axes=dgp_vmap,
        )
        dgp_output = dgp_batch_fn(*bound.arguments.values())
        data_store[scenario.simkey] = dgp_output

    return data_store
