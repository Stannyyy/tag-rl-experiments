import pytest

def test_math_works():
    assert 1 + 1 == 2

@pytest.mark.parametrize("a,b,expected", [(1, 2, 3), (0, 0, 0), (-1, 1, 0)])
def test_add(a, b, expected):
    assert a + b == expected
