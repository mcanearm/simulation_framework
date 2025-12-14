from jax._src.pjit import JitWrapped
import numpy as np
from typing import Union
from src.decorators import DGP
from src.utils import (
    DiskDict,
    bind_arguments,
    create_vmap_signature,
    generate_scenarios,
)
from collections.abc import MutableMapping
from pathlib import Path
import jax
import tqdm
import logging

from functools import singledispatch

logger = logging.getLogger(__name__)


@singledispatch
def generate_data(
    prng_key: jax.typing.ArrayLike,
    dgp_param_map: list[tuple[DGP | JitWrapped, dict]],
    n_sims: int,
    simulation_dir: Union[str, Path, None] = None,
    sequential_params: bool = False,
    allow_cache: bool = True,
) -> Union[MutableMapping[str, jax.typing.ArrayLike], dict]:
    data_gen_key = prng_key
    if simulation_dir is not None:
        # add n_sims to the output directory to ensure that different simulation sizes do not overwrite
        # each other from run to run
        output_dir = Path(simulation_dir)
        data_store = DiskDict(output_dir / "data", allow_cache=allow_cache)  # creates dir if it doesn't exist
    else:
        # use a plain dictionary if no data directory is provided
        data_store = dict()

    # iterate to start via generated data; once all data is generated, fit
    # each method on each dataset as it is generated.

    scenarios = [
        scenario
        for dgp_fn, param_set in dgp_param_map
        for scenario in generate_scenarios(
            dgp_fn, param_set, sequential=sequential_params
        )
    ]
    logger.info(f"{len(scenarios)} scenarios generated.")

    for scenario in tqdm.tqdm(scenarios, unit="datasets"):
        if scenario.simkey in data_store:
            logger.info(
                f"Data for scenario {scenario.simkey} already exists; skipping generation."
            )
            continue

        try:
            data_gen_key, _ = jax.random.split(prng_key, 2)
        # if attributeError, this is a numpy RNG style and we keep going
        except TypeError:
            pass

        dgp_fn, dgp_args = scenario
        dgp_output = _execute(data_gen_key, dgp_fn, dgp_args, n_sims)
        # overwrite the data_gen key on each iteration so that the data
        # is different on the next split.
        data_store[scenario.simkey] = dgp_output

    return data_store


@singledispatch
def _execute(rng: jax.typing.ArrayLike, dgp_fn, dgp_args, n_sims):
    sim_gen_keys = jax.random.split(rng, n_sims)
    dgp_argset = {dgp_fn._key_param_name: sim_gen_keys, **dgp_args}

    bound = bind_arguments(dgp_fn, **dgp_argset)
    dgp_vmap = create_vmap_signature(dgp_fn._key_param_name, bound)
    dgp_batch_fn = jax.vmap(
        dgp_fn,
        in_axes=dgp_vmap,
    )
    dgp_output = dgp_batch_fn(*bound.arguments.values())
    return dgp_output


@_execute.register
def _np_execute(rng: np.random.Generator, dgp_fn, dgp_args, n_sims: int):
    bound = bind_arguments(dgp_fn, **dgp_args)
    dgp_output = [dgp_fn(rng, *bound.arguments.values()) for _ in range(n_sims)]

    stacked_outputs = tuple(
        [
            np.stack([getattr(item, output_val) for item in dgp_output], axis=0)
            for output_val in dgp_fn.output
        ]
    )

    dgp_output = dgp_fn.output_class(*stacked_outputs)
    return dgp_output
