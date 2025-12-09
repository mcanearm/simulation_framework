import dill
import jax
from jax import numpy as jnp
import pytest

from src.utils import DiskDict, function_timer, key_to_str, simulation_grid
from tests.conftest import norm_data, ols, ols_data, ridge


def test_timer():
    with function_timer() as timer:
        _ = sum(range(100))
    assert timer.elapsed_time >= 0


@pytest.mark.parametrize("n", [50, [50, 100]])
@pytest.mark.parametrize("p", [10, [5, 10, 20]])
@pytest.mark.parametrize("alpha", [0.1, [0.1, 1.0]])
def test_simulation_grid(n, p, alpha):
    # Note here that this particular combination is invalid; that may or may
    # not be worth checking
    scenarios = simulation_grid(
        dgps=(norm_data, {"n": n, "p": p}),
        methods=[(ridge, {"alpha": alpha}), (ols, {})],
    )

    if isinstance(n, int):
        n = [n]
    if isinstance(p, int):
        p = [p]
    if isinstance(alpha, float):
        alpha = [alpha]

    assert len(scenarios.dgp) == len(n) * len(p)
    assert len(scenarios.method) == len(alpha) + 1  # since OLS doesn't change
    assert len(scenarios) == len(scenarios.dgp) * len(scenarios.method)


def test_dict_saving(tmpdir):
    data = DiskDict(tmpdir)
    data["test_key"] = "test_value"

    assert (tmpdir / "test_key.pkl").exists()
    assert data["test_key"] == "test_value"


def test_dict_resume(tmpdir):
    my_data = {"a": 2, "b": 1}
    with open(tmpdir / "my_data.pkl", "wb") as f:
        dill.dump(my_data, f)

    data = DiskDict(tmpdir)
    assert data["my_data"] == my_data


def test_dgp_saving(tmpdir):
    data = DiskDict(tmpdir)
    key = jax.random.PRNGKey(0)
    X, y, beta = ols_data(key, n=50, p=5)

    data["ols_data_0"] = {"X": X, "y": y, "beta": beta}

    loaded = data["ols_data_0"]
    assert jnp.array_equal(loaded["X"], X)
    assert jnp.array_equal(loaded["y"], y)
    assert jnp.array_equal(loaded["beta"], beta)


def test_key_str(key):
    assert key_to_str(key) == "key=0-0"
