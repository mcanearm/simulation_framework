from src.decorators import DGP, Method
from src.constants import VALID_KEY_NAMES
import jax
from jaxtyping import PRNGKeyArray
import inspect
from functools import partial

# What's needed to run with a DGP and Method?
# 1) output of the DGP saved to disk
# 2) Reuse the data with multiple parameters on the methods
# 3) Allocate a "chain" of steps that we can run sequentially
# Solution: DataCatalog and MethodCatalog classes that can save/load data and methods; Method Catalog takes a data catalog as input,
# but it should operate like a dictionary


def run_methods(
    rng_key: PRNGKeyArray,
    dgp: DGP,
    method: Method,
    n_sims: int = 10,
    **params,
):
    """
    Run a fully vectorized simulation setup, given the DGP and the method. Note here that for each array of parameters, we need to
    vectorize over them and somehow work out the output dimensionality. I think we can use Xarray for this.
    """
    pass
