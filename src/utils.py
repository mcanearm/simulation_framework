from collections.abc import MutableMapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import time
from typing import Union, Any

import dill
from jaxtyping import PRNGKeyArray

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


class Scenario(object):
    def __init__(self, fn: Union[Method, DGP], param_set):
        self.fn = fn
        self.param_set = param_set

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


@dataclass
class SimulationScenario(object):
    dgp: list[Scenario]
    method: list[Scenario]

    def __len__(self):
        return len(self.dgp) * len(self.method)


def simulation_grid(
    dgps: tuple[DGP, dict] | list[tuple[DGP, dict]],
    methods: tuple[Method, dict] | list[tuple[Method, dict]],
):
    """
    Get unique function calls for each of the data generating process and the methods.
    """

    if isinstance(dgps, tuple):
        dgps = [dgps]
    if isinstance(methods, tuple):
        methods = [methods]

    fn_calls = [
        [
            Scenario(method, param_set)
            for method, params in fn_set
            for param_set in get_arg_combinations(params)
        ]
        for fn_set in [dgps, methods]
    ]

    return SimulationScenario(*fn_calls)
