from time import time
from pathlib import Path


# def hashkey(key: PRNGKeyArray) -> str:
#     arr = np.asarray(key).tobytes()
#     hash = hashlib.sha1(arr).hexdigest()[:12]
#     return has


class function_timer(object):
    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time


class DiskDict(dict):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = Path(data_dir)

    def __getitem__(self, key):
        return super().__getitem__(key)

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


def simulation_grid(param_dict):
    pass
