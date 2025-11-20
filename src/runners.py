from src.decorators import MetadataWrapper
from functools import partial, update_wrapper
from itertools import product
import inspect
from collections import namedtuple
import numpy as np


class SimRunner(object):
    def __init__(
        self,
        data_generator: MetadataWrapper,
        method: MetadataWrapper,
        sim_params: dict[str, dict[str, object]],
        combinations: str = "all",
    ):
        self.data_generator = data_generator
        self.method = method
        # tie the parameters to the name
        assert isinstance(data_generator, MetadataWrapper), (
            "Data generator must use the DataGenerator class."
        )
        assert isinstance(method, MetadataWrapper), "Method must use the Model class."

        self.method_params = sim_params.get(method.label)
        self.data_gen_params = sim_params.get(data_generator.label)

        if True:
            # if combinations == "all":
            self.key_order = (*self.data_gen_params, *self.method_params)
            scenario_class = namedtuple("scenario", self.key_order)
            self.scenarios = [
                scenario_class(*args)
                for args in product(
                    *self.data_gen_params.values(), *self.method_params.values()
                )
            ]

    def __call__(self, N_sims, parallel=False):
        for scenario in self.scenarios:
            generator_params = {
                k: v for k, v in scenario._asdict().items() if k in self.data_gen_params
            }
            method_params = {
                k: v for k, v in scenario._asdict().items() if k in self.method_params
            }

            scenario_result = []
            for _ in range(N_sims):
                data = self.data_generator(**generator_params)

                # How to ensure that the data generator passes in the correct values
                # to the modelling function (in an arbitrary way)
                method_signature = inspect.signature(self.method)
                for param in data._asdict():
                    if param in method_signature.parameters:
                        method_params[param] = data._asdict()[param]
                result = self.method(**method_params)
                scenario_result.append(result)
            scenario_result = np.stack(scenario_result)
            yield scenario_result
