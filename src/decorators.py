from typing import Callable
from functools import update_wrapper
from collections import namedtuple


class MetadataWrapper(object):
    """
    This wrapper turns a normal function into a callable that provides an output class, which we implement
    strictly as a named tuple based on the wrapper label.
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
        elif self.output_class:
            return self.output_class(function_output)
        else:
            return function_output


# DataGeneratorClass
"""
Data generator class comes from a decorator that returns a callable object that; this callable takes the outputs
of the normal data generating process BUT returns a file system backed dictionary of the stored data.
"""


def _callable_wrap(label, output=None):
    def outer(fn: Callable) -> MetadataWrapper:
        return MetadataWrapper(fn, label, output)

    return outer


"""
Next, the model class needs to take an input dictionary of the scenarios with the data, but again, this input is 
a Filesystem backed dictionary. The model will then take the data from each input scenario and apply the models
to it, identified by the label and parameters. 
"""
# The "model" class should take a general "output" structure from a data generating
# process and tie it directly to its inputs. Then, the final outputs should have an Xarray
# output that includes the input parameters as a long array that created it...


model = data_generator = _callable_wrap
