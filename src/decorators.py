from typing import Callable
from functools import update_wrapper
import inspect
from collections import namedtuple


class MetadataWrapper(object):
    """
    Direct class interface for use with data generation objects.
    """

    def __init__(self, fn: Callable, label: str, output=None):
        self.fn = fn
        update_wrapper(self, fn)
        self.__wrapped__ = fn
        self.output_class = (
            namedtuple(f"{label.replace(' ', '')}", output) if output else None
        )
        self.label = label

    def __call__(self, *args, **kwargs):
        function_output = self.fn(*args, **kwargs)
        if isinstance(function_output, tuple) and self.output_class:
            return self.output_class(*function_output)
        else:
            return function_output


def _callable_wrap(label, output=None):
    def outer(fn: Callable) -> MetadataWrapper:
        return MetadataWrapper(fn, label, output)

    return outer


model = data_generator = _callable_wrap
