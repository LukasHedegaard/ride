import functools
import inspect
import math
import re
from argparse import Namespace
from contextlib import contextmanager
from operator import attrgetter
from typing import Any, Collection, Dict, Set, Union

from pytorch_lightning.utilities.parsing import AttributeDict

AttributeDictOrDict = Union[AttributeDict, Dict[str, Any]]


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def attributedict(attributedict_or_dict: AttributeDictOrDict) -> AttributeDict:
    """If given a dict, it is converted it to an argparse.AttributeDict. Otherwise, no change is made"""
    if isinstance(attributedict_or_dict, dict):
        return AttributeDict(**attributedict_or_dict)
    else:
        return attributedict_or_dict


def to_dict(d):
    if type(d) == Namespace:
        return vars(d)
    else:
        return dict(d)


def merge_attributedicts(*args):
    assert len(args) > 1
    acc = to_dict(args[0])
    for a in args[1:]:
        acc = {**acc, **to_dict(a)}
    return attributedict(acc)


def some(self, attr: str):
    try:
        a = attrgetter(attr)(self)
        return a is not None
    except Exception:
        return False


def some_callable(self, attr: str, min_num_args=0, max_num_args=math.inf):
    try:
        fn = attrgetter(attr)(self)
        if not callable(fn):
            return False
        num_args = len(inspect.getfullargspec(fn).args)
        return min_num_args <= num_args and num_args <= max_num_args
    except Exception:
        return False


def get(self, attr: str):
    try:
        a = attrgetter(attr)(self)
        return a
    except KeyError:
        return None


def differ_and_exist(a, b):
    return a is b and a is not None


def missing(self, attrs: Collection[str]) -> Set[str]:
    return {a for a in attrs if not some(self, a)}


def missing_or_not_in_other(
    first, other, attrs: Collection[str], must_be_callable=False
) -> Set[str]:
    some_ = some_callable if must_be_callable else some
    return {
        a
        for a in attrs
        if not some_(first, a) or differ_and_exist(get(first, a), get(other, a))
    }


def name(any):
    if hasattr(any, "__name__"):
        return any.__name__
    else:
        return any.__class__.__name__


def prefix_keys(prefix: str, dictionary: Dict) -> Dict:
    return {f"{prefix}{k}": v for k, v in dictionary.items()}


def camel_to_snake(s: str) -> str:
    """Convert from camel-case to snake-case
    Source: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


@contextmanager
def temporary_parameter(obj, attr, val):
    prev_val = rgetattr(obj, attr)
    rsetattr(obj, attr, val)
    yield obj
    rsetattr(obj, attr, prev_val)
