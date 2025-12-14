from collections.abc import MutableMapping
from dataclasses import dataclass
from inspect import BoundArguments
from itertools import product
from pathlib import Path
from time import time
import logging
import jax

import dill
from jaxtyping import PRNGKeyArray

from src.decorators import MetadataCaller
from src.decorators import DGP, Method

logger = logging.getLogger(__name__)


def key_to_str(key: PRNGKeyArray) -> str:
    """
    Convert a jax key to a string representation for use in filepaths. Works
    with both legacy and new jax key formats.
    """
    try:
        key_param = "-".join([str(i) for i in key.tolist()])
    except AttributeError:
        key_data = jax.random.key_data(key)
        key_param = "-".join([str(i) for i in key_data])
    return f"key={key_param}"


class function_timer(object):
    """
    Context manager for timing a block of code.

    Usage:
        with(function_timer() as timer):
            # code block to time

        print(f"Elapsed time: {timer.elapsed_time} seconds")
    """

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time


class DiskDict(MutableMapping):
    """
    Dictionary-like object that stores objects via dill on disk. Each key is stored as a separate .pkl file, and
    by default, caching is used with a standard dictionary to avoid repeated disk reads.

    Args:
        data_dir (str | Path): Directory to store the pickled objects.
        allow_cache (bool): Whether to cache loaded objects in memory. Default is True.

    Returns:
        DiskDict: A dictionary-like object that persists data on disk.
    """

    def __init__(self, data_dir, allow_cache=True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.allow_cache = allow_cache
        self.cache = {}

    def __getitem__(self, key):
        filepath = self.data_dir / f"{key}.pkl"
        if filepath.exists():
            if key in self.cache:
                result = self.cache[key]
            else:
                with open(filepath, "rb") as f:
                    result = dill.load(f)
                if self.allow_cache:
                    self.cache[key] = result
            return result
        else:
            raise KeyError(f"Key {key} not found in DiskDict at {filepath}.")

    def __setitem__(self, key, value) -> None:
        filepath = self.data_dir / f"{key}.pkl"
        with open(filepath, "wb") as f:
            dill.dump(value, f)
        if self.allow_cache:
            self.cache[key] = value

    def __delitem__(self, key) -> None:
        filepath = self.data_dir / f"{key}.pkl"
        if filepath.exists():
            filepath.unlink()
        if key in self.cache:
            del self.cache[key]

    def __iter__(self):
        for file in self.data_dir.glob("*.pkl"):
            yield file.stem

    def __len__(self) -> int:
        return len(list(self.data_dir.glob("*.pkl")))


def get_arg_combinations(params: dict):
    """
    Given a dict of lists pairing, return all combinations of the method with the parameter grid.
    """
    for k, v in params.items():
        if not isinstance(v, list):
            params[k] = [v]

    combos = [
        {k: v for k, v in zip(params.keys(), param_combination)}
        for param_combination in product(*params.values())
    ]
    return combos


@dataclass
class Scenario(object):
    fn: MetadataCaller
    param_set: dict

    @property
    def filename(self) -> str:
        return f"{self.simkey}.pkl"

    @property
    def simkey(self) -> str:
        param_str = "_".join([f"{k}={v}" for k, v in self.param_set.items()])
        return f"{self.fn.label}_{param_str}"

    def __repr__(self):
        return f"Scenario(fn={self.fn.label}, params={self.param_set})"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        yield from [self.fn, self.param_set]


def generate_scenarios(
    fn: DGP | Method, param_grid: dict[str, list[object]], sequential: bool = False
) -> list[Scenario]:
    """
    Generate simulation scenarios based on the provided parameter grid. By
    default, this returns a factorial set combination of scenarios, but
    can also treat each parameter as sequential (i.e., same length lists).

    Args:
        fn (callable): The function (DGP or Method) for which to generate scenarios
        param_grid (dict): A dictionary where keys are parameter names and values are lists of parameter values.
    """
    if sequential:
        logger.debug("Generating scenarios sequentially.")
        try:
            param_sets = [
                {k: v for k, v in zip(param_grid.keys(), param_val)}
                for param_val in zip(*param_grid.values())
            ]
            scenarios = [Scenario(fn, param_set) for param_set in param_sets]

        except IndexError:
            raise IndexError(
                "All parameters provided must be the same length for sequential scenario generation."
            )

    else:
        logger.debug("Generating scenarios in factorial manner.")
        scenarios = [
            Scenario(fn, param_set) for param_set in get_arg_combinations(param_grid)
        ]
    return scenarios


def get_params_from_scenario_keystring(keystring, keystr_type=None) -> dict:
    """
    Work back from a scenario keystring to the parameter dictionary. If the
    keystring contains a double underscore, it is assumed to be of the form
    {data}__{method}, and both parts are parsed separately and combined into a
    single dict. The label for data and method are returned as well.

    Args:
        keystring (str): The scenario keystring to parse.
        keystr_type (str | None): This never needs to be provided; it is used
        recursively to label the data and method parts separately.
    Returns:
        dict: A dictionary of parameter names and values extracted from the
        keystring.
    """
    if "__" in keystring:
        data_keystr, method_keystr = keystring.split("__")
        params = [
            get_params_from_scenario_keystring(part, type)
            for part, type in zip([data_keystr, method_keystr], ["data", "method"])
        ]
        param_dict = {
            k: v for param_dict in params for k, v in param_dict.items() if param_dict
        }
        return param_dict
    else:
        param_strs = keystring.split("_")
        label = {keystr_type: param_strs[0]}
        param_strs = param_strs[1:]
        if not param_strs:
            return label
        else:
            param_strs = [p for p in param_strs if "=" in p]
            param_dict = {
                k: v for param in param_strs for k, v in [param.split("=")] if param
            }
            return {**label, **param_dict}


def bind_arguments(fn: DGP | Method, *args, **kwargs) -> BoundArguments:
    """
    Bind the provided args and kwargs to the function signature, applying defaults
    where necessary. This is a helper function to prepare for vmap signatures.

    Args:
        fn (callable): The function whose signature to bind.
        *args: Positional arguments to bind.
        **kwargs: Keyword arguments to bind.
    Returns:
        BoundArguments: The bound arguments object.
    """
    match_args = {k: v for k, v in kwargs.items() if k in fn.sig.parameters}
    sig = fn.sig
    bound = sig.bind_partial(*args, **match_args)
    bound.apply_defaults()

    return bound


def create_vmap_signature(
    vmap_args: str | list[str], bound_args: BoundArguments
) -> tuple:
    """
    Given an argument name or a list of argument names to vmap over and the
    set of bound arguments, create a vmap signature that is simply a tuple
    sequence of 0 (for vmap) and None (for no vmap). This implicitly assumes
    that the FIRST dimension is the thing to vmap over, but this is how
    all functions work in the framework. Custom vmap logic can be nested,
    however, so this is sufficient for most use cases.

    Args:
        vmap_args (str | list[str]): Argument name(s) to vmap over.
        bound_args (BoundArguments): The bound arguments of the function.
    Returns:
        tuple: A tuple of in_axes specifications for jax.vmap.
    """
    if isinstance(vmap_args, str):
        vmap_args = [vmap_args]
    return tuple(
        [
            0 if param.name in vmap_args else None
            for param in bound_args.signature.parameters.values()
        ]
    )
