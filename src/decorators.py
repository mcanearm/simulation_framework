from collections import namedtuple
import inspect
from functools import update_wrapper

from jaxtyping import Array

from src.constants import VALID_KEY_NAMES


class MetadataCaller(object):
    """
    This is the workhorse class for all decorated functions in the simulation
    study framework, but is never used directly. Instead, we utilize it as a
    catch all since the specific types of functions (DGPs, Methods, Evaluators, Plotters)
    have no functional differences for now, but could eventually be subclassed
    to modify this existing logic.

    The main purpose of this class is wrap an existing function, preserving
    its docstring and function signature, but output a namedtuple with
    the specified output fields. When this is done, this serves as a contract
    by which other methods can align inputs and outputs.

    These classes are also not invoked directly (though they can be if desired).
    Instead, the intended use is to use the decorator interface as a factory
    for these methods.

    Args:
        fn (callable): The function to wrap.
        label (str): The label for the function (used in namedtuple).
        output (list[str]): The output field names for the namedtuple.
    """

    def __init__(self, fn, label, output):
        self.fn = fn
        self.label = label
        self.output = output

        update_wrapper(self, fn)
        self.sig = inspect.signature(fn)
        self.output_class = namedtuple(self.label, self.output)

        # NOTE: there is shared logic here with the DGP decorator. If the
        # method is a DGP, the key MUST be first. Otherwise, the key does not
        # have to be provided, but if it is, it also MUST be first. This
        # slightly weird nesting logic should be refactored to avoid
        # repetition, but it's unclear exactly how to do so elegantly.

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

    def __call__(self, *args, **kwargs) -> Array | tuple[Array]:
        # if a data directory is provided, assume it was provided to save/load data
        result = self.fn(*args, **kwargs)
        if not isinstance(result, tuple):
            return self.output_class(result)
        else:
            return self.output_class(*result)


class DGP(MetadataCaller):
    """
    DGP class to mark data generating processes in simulation studies.
    Functionally identical to MetadataCaller, but exists for clarity and
    for future subclassing.
    """

    def __init__(self, fn, label, output):
        super().__init__(fn, label, output)

    def __repr__(self) -> str:
        return f"<DGP: {self.label}, output: {self.output}>"


class Evaluator(MetadataCaller):
    """
    Evaluator class to mark evaluation functions in simulation studies.
    Functionally identical to MetadataCaller, but exists for clarity and
    for future subclassing.
    """

    def __init__(self, fn, label, output):
        super().__init__(fn, label, output)

    def __repr__(self) -> str:
        return f"<Evaluator: {self.label}, output: {self.output}>"


class Method(MetadataCaller):
    """
    Method class to denote method functions in simulation studies.
    Functionally identical to MetadataCaller, but exists for clarity and
    for future subclassing.
    """

    def __init__(self, fn, label, output):
        super().__init__(fn, label, output)

    def __repr__(self) -> str:
        return f"<Method: {self.label}, output: {self.output}>"


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

    def outer(fn):
        inner_label = label or fn.__name__
        return Method(fn=fn, label=inner_label, output=output)

    return outer


def evaluator(output, label=None):
    """
    Enforce an evaluator contract for simulation studies. Designed to take the
    outputs of methods and data generating processes and calculate evaluation
    metrics.
    """

    def outer(fn):
        inner_label = label or fn.__name__
        return Evaluator(fn=fn, label=inner_label, output=output)

    return outer
