import pytest
from main import minhash_bands


@pytest.mark.parametrize("n, s", [(100, 0.8), (65, 0.6)])
def test_minhash(n, s):
    b, r = minhash_bands(n, s)
    assert 1-(1-(s**r))**b >= 0.5