from functools import update_wrapper
import inspect
from src.constants import VALID_KEY_NAMES
from collections import namedtuple


class MetadataCaller(object):
    def __init__(self, fn, label, output):
        self.fn = fn
        self.label = label
        self.output = output
        self.output_class = (
            namedtuple(label, output) if len(output) > 1 else lambda x: x
        )

        update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if not isinstance(self.output, (list, tuple)):
            return self.output_class(self.fn(*args, **kwargs))
        else:
            return self.output_class(*self.fn(*args, **kwargs))


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

    Valid key parameter names are: "key", "rng_key", "prng_key", "PRNGKey"
    """

    def outer(fn):
        sig = inspect.signature(fn)
        params = sig.parameters

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
