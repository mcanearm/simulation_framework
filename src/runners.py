from src.decorators import MetadataWrapper
import inspect
from sklearn.model_selection import ParameterGrid
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import numpy as np


class SimRunner(object):
    """
    This is the initial attempt at a simulation runner, but it already feels too heavy. The data generation should
    probably be completely separate from the actual running of the data. In fact, I think we want an entire class dedicated
    to data generation, and one for running the models on various scenarios. The scenarios can be loaded from disk using a
    dictionary like interface.
    """

    def __init__(
        self,
        data_generator: MetadataWrapper,
        method: MetadataWrapper,
        sim_params: dict[str, dict[str, object]] = None,
        scenarios=None,
    ):
        self.data_generator = data_generator
        self.method = method
        # tie the parameters to the name
        assert isinstance(data_generator, MetadataWrapper), (
            "Data generator must use the DataGenerator class."
        )
        assert isinstance(method, MetadataWrapper), "Method must use the Model class."

        if sim_params:
            self.method_params = {
                f"__{method.label}_{k}": v
                for k, v in sim_params.get(method.label, {}).items()
            }
            self.data_gen_params = {
                f"__{data_generator.label}_{k}": v
                for k, v in sim_params.get(data_generator.label, {}).items()
            }

            # for now, only support factorial designs
            full_parameter_set = {**self.method_params, **self.data_gen_params}
            self.all_scenarios = ParameterGrid(full_parameter_set)
        elif scenarios:
            self.all_scenarios = scenarios
        else:
            assert False, "Must provide either sim_params or scenarios."

        # TODO: Add a filesystem backed dictionary where we will store the results
        # of data generating processes for each scenario
        self.filesystem_backed_dictionary = None

    def _parse_scenario_args(self, scenario):
        dgp_args = {
            k.replace(param_lab, ""): v
            for k, v in scenario.items()
            if k.startswith(param_lab := f"__{self.data_generator.label}_")
        }
        method_args = {
            k.replace(param_lab, ""): v
            for k, v in scenario.items()
            if k.startswith(param_lab := f"__{self.method.label}_")
        }
        return dgp_args, method_args

    def generate_data(self, scenario, N_sims):
        dgp_args, _ = self._parse_scenario_args(scenario)
        sim_data = self.data_generator(**dgp_args)
        return sim_data

    def _run_scenario(self, scenario, N_sims):
        dgp_args, method_args = self._parse_scenario_args(scenario)

        # TODO: the problem here is that I want to re-use the data that I generate
        # for multiple methods. So theoretically, all the data generation should happen
        # first and outside of this function, and then it should get passed into the method
        # bits
        for i in range(N_sims):
            sim_data = self.data_generator(**dgp_args)

            shared_params = set(sim_data._fields).intersection(
                inspect.signature(self.method.fn).parameters
            )

            method_inputs = {k: sim_data.__getattribute__(k) for k in shared_params}

            method_output = self.method(**method_inputs, **method_args)

            return method_output

    def run_simulations(self, N_sims=10, parallel=False):
        return [self._run_scenario(scenario, N_sims) for scenario in self.all_scenarios]


def pool_function(
    fn,
    ncores=1,
    nsims=100,
    pool_type="thread",
    *args,
    **kwargs,
):
    """
    Run a function, pooled, using concurrent futures.
    """

    if ncores > 1:
        pool_class = (
            ThreadPoolExecutor if pool_type == "thread" else ProcessPoolExecutor
        )

        spawn_seeds = np.random.default_rng().spawn(ncores)

        with pool_class(max_workers=ncores) as executor:
            futures = {executor.submit(fn, *args, **kwargs) for _ in range(nsims)}
            results = [i.result() for i in as_completed(futures)]
    else:
        results = [fn(*args, **kwargs) for _ in range(nsims)]

    return results


def simple_pool_fn(x):
    return x**2 + np.random.normal(0, 1)


if __name__ == "__main__":
    pool_function(
        simple_pool_fn, ncores=4, nsims=10, pool_type="process", x=np.arange(1000)
    )
