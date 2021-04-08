from ride import info


def test_info():
    assert type(info.__version__) == str
    assert type(info.__author__) == str
    assert type(info.__author_email__) == str
    assert type(info.__license__) == str
    assert type(info.__copyright__) == str
    assert type(info.__homepage__) == str
    assert type(info.__docs__) == str
