import inspect
import textwrap

import jax
import pytest
from jax import numpy as jnp

from src.constants import VALID_KEY_NAMES
from src.decorators import dgp, method
from src.utils import simulation_grid
from tests.conftest import norm_data, ridge


def test_dgp(key):
    samples = norm_data(key, n=10, p=5)
    assert samples.shape == (10, 5)
    assert norm_data.label == "normal"


@pytest.mark.parametrize("fn", [norm_data, ridge])
def test_docstring_preservation(fn):
    tmp = inspect.getdoc(fn)
    assert tmp == "test docstring"


def test_vmappable(key):
    key_array = jax.random.split(key, 5)
    assert jax.vmap(norm_data, in_axes=(0,))(key_array).shape == (5, 100, 10)


def test_jitable(key):
    # you can't change the n or p, since shapes are really important for jit compilation I suppose. But this
    # is a JAX thing, not our package, and you can at least JIT any DGP function
    simple_norm = jax.jit(norm_data, static_argnames=("n", "p"))
    assert simple_norm(key, n=10, p=3) is not None


def test_wrong_key_name():
    with pytest.raises(ValueError):

        @dgp(output="test")
        def bad_fn(wrong_key_name, n=10):
            return jnp.zeros((n, 1))


@pytest.mark.parametrize("deco_class", [dgp, method])
@pytest.mark.parametrize("valid_key_name", VALID_KEY_NAMES)
def test_right_key_wrong_place(deco_class, valid_key_name):
    with pytest.raises(ValueError):
        my_fn = f"""
        @{deco_class.__name__}(output="test")
        def bad_fn(n, {valid_key_name}):
            return jnp.zeros((n, 1))
        """
        exec(textwrap.dedent(my_fn))


def test_grid_call(key):
    pass
