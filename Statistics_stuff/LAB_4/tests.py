import pytest
from tasks import gen_hash, task2


#Its hard to make test for those tasks because i'm using many random functions inside task so the results are estimated

@pytest.mark.parametrize("m, bitstr", [(10, "10101010"), (1, "10101010")])
def test_gen_hash(m, bitstr):
    assert len(gen_hash(m)(bitstr)) != 0


@pytest.mark.parametrize("n", [10])
def test_task2(n):
    assert len(task2(n)) != 0


@pytest.mark.parametrize("n", [1])
def test_task2(n):
    assert task2(n) == False