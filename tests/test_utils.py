from src.utils import function_timer, DiskDict, simulation_grid
from tests.conftest import norm_data, ridge, ols
from src.decorators import dgp
import jax
from jax import numpy as jnp
import dill


@dgp(output=["X", "y", "beta"])
def ols_data(prng_key, n=100, p=10):
    k1, k2 = jax.random.split(prng_key, 2)
    X = jax.random.normal(k1, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    y = X @ true_beta + jax.random.normal(k2, shape=(n,)) * 0.5
    return X, y, true_beta


def test_timer():
    with function_timer() as timer:
        _ = sum(range(100))
    assert timer.elapsed_time >= 0


def test_simulation_grid():
    scenarios = simulation_grid(
        dgps=(norm_data, {"n": [50, 100], "p": [5, 10]}),
        methods=[(ridge, {"alpha": [0.1, 1.0]}), (ols, {})],
    )

    assert len(scenarios.dgp[0][1]) == 4  # test 4 parameter combos
    assert len(scenarios.method) == 2  # test 2 methods output


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
