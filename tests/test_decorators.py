import pytest
from src.decorators import data_generator
import numpy as np
import inspect


@pytest.fixture
def rnorm():
    @data_generator("rnorm")
    def _rnorm(n):
        """
        test docstring
        """
        return np.random.normal(size=n)

    return _rnorm


def test_rnorm(rnorm):
    n = 10

    samples = rnorm(n)
    assert len(samples) == n
    assert rnorm.name == "rnorm"


def test_docstring_preservation(rnorm):
    tmp = inspect.getdoc(rnorm)
    assert tmp == "test docstring"
