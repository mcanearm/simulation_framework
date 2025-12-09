from collections.abc import MutableMapping
from dataclasses import dataclass
from inspect import BoundArguments
from itertools import product
from pathlib import Path
from time import time
from typing import Any

import dill
from jaxtyping import PRNGKeyArray

from decorators import MetadataCaller
from src.decorators import DGP, Method


def key_to_str(key: PRNGKeyArray) -> str:
    key_param = "-".join([str(i) for i in key.tolist()])
    return f"key={key_param}"


class function_timer(object):
    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time


class DiskDict(MutableMapping):
    def __init__(self, data_dir, allow_cache=True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key):
        filepath = self.data_dir / f"{key}.pkl"
        if filepath.exists():
            with open(filepath, "rb") as f:
                result = dill.load(f)
            return result
        else:
            raise KeyError(f"Key {key} not found in DiskDict at {filepath}.")

    def __setitem__(self, key, value) -> None:
        filepath = self.data_dir / f"{key}.pkl"
        with open(filepath, "wb") as f:
            dill.dump(value, f)

    def __delitem__(self, key) -> None:
        filepath = self.data_dir / f"{key}.pkl"
        if filepath.exists():
            filepath.unlink()

    def __iter__(self):
        for file in self.data_dir.glob("*.pkl"):
            yield file.stem

    def __len__(self) -> int:
        return len(list(self.data_dir.glob("*.pkl")))


def get_arg_combinations(params):
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
    params.values()
    return combos


def get_scenario_params(scenario_key_str: str) -> tuple[str, dict[str, Any]]:
    param_strs = scenario_key_str.split("_")
    param_dict = {k: v for param in param_strs[1:] for k, v in [param.split("=")]}
    return param_strs[0], param_dict


def construct_scenarios(fn, param_grid):
    return [Scenario(fn, param_set) for param_set in get_arg_combinations(param_grid)]


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


def bind_arguments(fn: DGP | Method, *args, **kwargs) -> BoundArguments:
    match_args = {k: v for k, v in kwargs.items() if k in fn.sig.parameters}
    sig = fn.sig
    bound = sig.bind_partial(*args, **match_args)
    bound.apply_defaults()

    # skip the prng key argument for vmap
    return bound


def create_vmap_signature(
    vmap_args: str | list[str], bound_args: BoundArguments
) -> tuple:
    if isinstance(vmap_args, str):
        vmap_args = [vmap_args]
    return tuple(
        [
            0 if param.name in vmap_args else None
            for param in bound_args.signature.parameters.values()
        ]
    )
