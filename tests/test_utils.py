import pytest
import dill
import jax
from jax import numpy as jnp

from src.utils import DiskDict, function_timer, key_to_str, generate_scenarios
from tests.conftest import ols_data


def test_timer():
    with function_timer() as timer:
        _ = sum(range(100))
    assert timer.elapsed_time >= 0


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


@pytest.mark.parametrize(
    "sequential,answer", [(False, 4), (True, 2)], ids=["factorial", "sequential"]
)
def test_scenario_generation(sequential, answer):
    param_grid = {"n": [100, 200], "p": [5, 10]}
    scenarios = generate_scenarios(ols_data, param_grid, sequential=sequential)
    assert len(scenarios) == answer
