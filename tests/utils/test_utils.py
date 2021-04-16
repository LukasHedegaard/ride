from collections import namedtuple

from ride.utils.utils import is_shape


def test_is_shape():
    # OK
    assert is_shape(42)  # int
    assert is_shape([42, 13])  # list of ints
    assert is_shape((42,))  # tuple of ints

    MyShape = namedtuple("MyShape", ["x", "y"])
    assert is_shape(MyShape(42, 13))  # namedtuple of ints

    # Not OK
    class MyClass:
        ...

    assert not is_shape(MyClass)  # regular class
    assert not is_shape(42.0)  # float
    assert not is_shape("42")  # str
    assert not is_shape((42, "13"))  # tuple of mixed types
