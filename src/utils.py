from time import time
from pathlib import Path
from src.decorators import DGP, Method
from itertools import product
from collections import namedtuple
from collections.abc import MutableMapping
import dill


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

    # if isinstance(self.data_dir, Path):
    #         filename, argdict = self.__generate_filename(*args, **kwargs)
    #         filepath = self.data_dir / filename  # type: ignore
    #         if filepath.exists():
    #             # load data
    #             with open(filepath, "rb") as f:
    #                 result = dill.load(f)
    #             self._cache_hits += 1
    #             self._last_call = {
    #                 **argdict,
    #                 "filepath": filepath,
    #             }
    #         else:
    #             result = self.fn(*args, **kwargs)
    #             with open(filepath, "wb") as f:
    #                 dill.dump(result, f)
    #             self._last_call = {**argdict, "filepath": filepath}
    #     else:
    #         # if no data directory, just call the function and exit


# def __generate_filename(self, *args, **kwargs):
#         bound = self.sig.bind(*args, **kwargs)
#         bound.apply_defaults()
#         argdict = dict(bound.arguments)
#         key_hash = None
#         if False:  # self._key_param_name or self._key_param_name in argdict:
#             key_value = argdict.pop(self._key_param_name)
#             key_hash = hashkey(key_value)
#             param_based_path = "_".join(
#                 [f"{k}={v}" for k, v in argdict.items() if not isinstance(v, Array)]
#             )
#         else:
#             param_based_path = "_".join(
#                 [f"{k}={v}" for k, v in argdict.items() if not isinstance(v, Array)]
#             )
#         param_path = f"{param_based_path}" if key_hash else param_based_path

#         return f"{self.label}_{self._role}_{param_path}.pkl", argdict


def get_arg_combinations(params):
    """
    Given a dict of lists pairing, return all combinations of the method with the parameter grid.
    """
    return [
        {k: v for k, v in zip(params.keys(), param_combination)}
        for param_combination in product(*params.values())
    ]


# NOTE: Starting point of directory structure is the hash of the starting jax key
# prior to splitting


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

    dgp_kwarg_sets = [(dgp, get_arg_combinations(params)) for dgp, params in dgps]
    method_kwarg_sets = [
        (method, get_arg_combinations(params)) for method, params in methods
    ]
    output_class = namedtuple("SimulationScenario", ["dgp", "method"])

    return output_class(dgp_kwarg_sets, method_kwarg_sets)
