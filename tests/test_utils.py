from src.utils import function_timer
from src.utils import simulation_grid


def test_timer():
    with function_timer() as timer:
        _ = sum(range(100))
    assert timer.elapsed_time >= 0


def test_dict_saving(tmpdir):
    pass
    # @dgp(output=["X", "y", "beta"])
    # def ols_data(prng_key, n=100, p=10):
    #     k1, k2 = jax.random.split(prng_key, 2)
    #     X = jax.random.normal(k1, shape=(n, p))
    #     true_beta = jnp.arange(1, p + 1)
    #     y = X @ true_beta + jax.random.normal(k2, shape=(n,)) * 0.5
    #     return X, y, true_beta

    # new_key, _ = jax.random.split(key)

    # output = ols_data(key)
    # output2 = ols_data(key)  # should load from disk
    # assert ols_data._cache_hits == 1

    # # we should hit the cache when using a new key
    # output3 = ols_data(new_key)
    # assert jax.numpy.not_equal(output3[0], output[0]).all()
    # assert ols_data._cache_hits == 1
