import pytest
from main import sim_bitstrings, set_jaccard, bag_jaccard, shingles, diameter, hamming_distance


@pytest.mark.parametrize("b,n,result", [('010', 1, ['000', '011', '110']), ('001', 1, ['000', '011', '101'])])
def test_sim_bitstrings(b, n, result):
    assert sim_bitstrings(b, n) == result


@pytest.mark.parametrize("x,y,result",
                         [({'hello', 'world'}, {'goodbye', 'world'}, 1 / 3), ({1, 2, 3}, set(), None)])
def test_set_jaccard(x, y, result):
    assert set_jaccard(x, y) == result


@pytest.mark.parametrize("x,y,result",
                         [('hello world', 'goodbye world', 1 / 3), ('hello world', '', 0)])
def test_bag_jaccard(x, y, result):
    assert bag_jaccard(x, y) == result


@pytest.mark.parametrize("s,k,result", [("The sky is blue and the sun is bright.", 4,
                                         {'and the sun is', 'The sky is blue', 'sky is blue and', 'is blue and the',
                                          'blue and the sun', 'the sun is bright.'})])
def test_shingles(s, k, result):
    assert shingles(s, k) == result


@pytest.mark.parametrize("S,d,result", [({'hello', 'world', 'cruel', 'world'}, hamming_distance,5)])
def test_diameter(S, d, result):
    assert diameter(S, d) == result
