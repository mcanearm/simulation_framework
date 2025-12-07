import hashlib
import inspect
from collections import namedtuple
from functools import update_wrapper
from pathlib import Path

import dill
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from src.constants import VALID_KEY_NAMES


class MetadataCaller(object):
    def __init__(self, fn, label, output):
        self.fn = fn
        self.label = label
        self.output = output

        update_wrapper(self, fn)
        self.sig = inspect.signature(fn)

        if set(self.sig.parameters.keys()).intersection(VALID_KEY_NAMES):
            first_item = list(self.sig.parameters.keys())[0]
            if first_item in VALID_KEY_NAMES:
                self._key_param_name = first_item
                self._requires_key = True
            else:
                raise ValueError(
                    f"Key argument in function signature is not valid. The FIRST function argument must be one of {VALID_KEY_NAMES}."
                )
        else:
            self._key_param_name = None
            self._requires_key = False

        # self.output_class = namedtuple(self.label, self.output)
        self.sig = inspect.signature(fn)

    def __call__(self, *args, **kwargs):
        # if a data directory is provided, assume it was provided to save/load data
        result = self.fn(*args, **kwargs)
        return result


class DGP(MetadataCaller):
    def __init__(self, fn, label, output):
        super().__init__(fn, label, output)
        self._role = "DGP"


class Method(MetadataCaller):
    def __init__(self, fn, label, output):
        super().__init__(fn, label, output)
        self._role = "Method"


def dgp(output, label=None):
    """
    Small decorator to enforce JAX-style generator contract for simulation studies. Exists to do two things:
    1) Mark the function as a DGP (with a relevant label for future steps)
    2) Ensure the first argument is a PRNG key

    Valid key parameter names are listed in VALID_KEY_NAMES in src/constants.py
    """

    def outer(fn):
        sig = inspect.signature(fn)
        params = sig.parameters

        # ensures that the first argument is a valid key name; required for data generating processes
        first_item = list(params.keys())[0]
        if first_item not in VALID_KEY_NAMES:
            raise ValueError(
                f"Key argument in function signature is not valid. The FIRST function argument must be one of {VALID_KEY_NAMES}."
            )
        inner_label = label or fn.__name__

        return DGP(fn=fn, label=inner_label, output=output)

    return outer


def method(output, label=None):
    """
    Enforce a method contract for simulation studies. Needs to take the output of a DGP as input
    """

    # Slightly different; if the prngkey is an argument, it MUST be first. However,
    # it is not required.

    def outer(fn):
        sig = inspect.signature(fn)

        if set(sig.parameters.keys()).intersection(VALID_KEY_NAMES):
            first_item = list(sig.parameters.keys())[0]
            if first_item not in VALID_KEY_NAMES:
                raise ValueError(
                    f"Key argument in function signature is not valid. The FIRST function argument must be one of {VALID_KEY_NAMES}."
                )

        inner_label = label or fn.__name__
        return Method(fn=fn, label=inner_label, output=output)

    return outer
