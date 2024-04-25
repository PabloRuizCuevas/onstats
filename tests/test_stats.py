from typing import Generator

import numpy as np
import pytest

from onstats.stats import (
    ath,
    avg,
    ma,
    normalize,
    raw,
    var,
    sum_f,
    wsum,
    wsum_p,
    ema,
    diff,
    cov_xy,
    delay,
    corr_xy,
    percentual_spread,
)
from onstats.util import isend, send
import pandas as pd


# Add more function names as needed
def generators_to_test():
    return [ma(3), ath(), raw(), avg()]


data_1d = [2, 3, 7, 2, 3]
solutions = [[2, 2.5, 4, 4, 4], [2, 3, 7, 7, 7], data_1d, [2, 2.5, 4.0, 3.5, 3.4]]


@pytest.fixture
def buff_1d_gen():
    return (i for i in data_1d)


@pytest.fixture
def b2d_gen():
    print("Cleanup after the test")
    data = np.array(
        [[2, 3], [4, 2], [3, 4], [2, 3], [1, 2], [1, 1], [1, 1], [1, 1], [1, 1]]
    )
    b_gen = (last for last in data)
    yield b_gen


@pytest.fixture
def gen_c():
    return (i for i in [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])


@pytest.mark.parametrize("generator", generators_to_test())
def test_indicators_ids(generator: Generator, b2d_gen):
    # Dynamically get the function from my_module based on func_name
    # if  id () in the list, python woudl reuse id inside the generator
    # very confusing
    a = send(b2d_gen, generator)
    id_set = set([id(i) for i in a])
    assert len(id_set) == 9, "All ids should be different"


def test_wrap(gen_c):
    # no funciona pq como el wsum usa el buffer... terrible error
    # pues no estas mandando el buffer bien modificado, o si
    # pero vaya, mal
    mapped = send(gen_c, sum_f(2, lambda x: np.maximum(x, 0)))
    np.testing.assert_array_equal(
        np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), mapped
    )


def test_var_pop():
    mapped = send(data_1d, var(2, ddof=0))
    rolling_std = pd.Series([2, 3, 7, 2, 3]).rolling(window=2).var(0).tolist()
    np.testing.assert_array_equal(mapped[1:], rolling_std[1:])


def test_var_sample():
    mapped = send(data_1d, var(2))
    rolling_std = pd.Series([2, 3, 7, 2, 3]).rolling(window=2).var(1).tolist()
    np.testing.assert_array_equal(mapped[1:], rolling_std[1:])


def test_var_sample_n():
    mapped = send(data_1d, var(10))
    sol = [0.0, 0.5, 7.0, 5.666666666666667, 4.300000000000001]
    np.testing.assert_array_equal(mapped, sol)
    mapped = send(data_1d, var(100))
    np.testing.assert_array_equal(mapped, sol)
    mapped = send(data_1d, var(200))
    np.testing.assert_array_equal(mapped, sol)


def test_percentual_spread():
    data = np.array([[1, 1, 1, 1, 2], [2, 3, 7, 2, 3]])
    mapped = send(data.T, percentual_spread())
    np.testing.assert_array_equal(mapped, [1.0, 2.0, 6.0, 1.0, 0.5])


def test_cov_same_as_std():
    data = np.array([[2, 3, 7, 2, 3], [2, 3, 7, 2, 3]])
    mapped = send(data.T, cov_xy(2, 0))
    rolling_std = pd.Series([2, 3, 7, 2, 3]).rolling(window=2).var(0).tolist()
    np.testing.assert_array_equal(mapped[1:], rolling_std[1:])


def test_cov():
    data = np.array([[1, 2], [2, 4], [4, 8], [8, 16], [16, 32]])
    mapped = send(data, cov_xy(5, 0))
    cov = np.cov(data.T, ddof=0)
    np.testing.assert_almost_equal(mapped[-1], cov[1][0])
    np.testing.assert_almost_equal(mapped[-1], cov[0][1])

    mapped = send(data, cov_xy(5))
    cov = np.cov(data.T)
    np.testing.assert_almost_equal(mapped[-1], cov[1][0])
    np.testing.assert_almost_equal(mapped[-1], cov[0][1])


def test_corr():
    data = np.array([[1, 2], [2, 4], [4, 8], [8, 16], [16, 32]])
    mapped = send(data, corr_xy(5))
    corr = np.corrcoef(data.T)  # , ddof=0
    np.testing.assert_almost_equal(mapped[-1], corr[1][0])
    np.testing.assert_almost_equal(mapped[-1], corr[0][1])

    data = np.array([[2, 3, 2, 2, 3], [2, 1, 7, -3, 3]])
    mapped = send(data.T, corr_xy(5, 1))
    corr = np.corrcoef(data)
    np.testing.assert_almost_equal(mapped[-1], corr[1][0])


def test_normalize():
    data = np.array([[2, 3, 7, 2, 3], [2, 3, 7, 2, 3]])
    mapped = send(data.T, normalize())


def test_delay():
    data = [2, 3, 7, 2, 3]
    mapped = send(data, delay(0))
    np.testing.assert_almost_equal(mapped, data)
    mapped = send(data, delay(1))
    np.testing.assert_almost_equal(mapped, [0, 2, 3, 7, 2])
    mapped = send(data, delay(2))
    np.testing.assert_almost_equal(mapped, [0, 0, 2, 3, 7])


def test_wsum():
    mapped = send(data_1d, wsum(2))
    series = pd.Series(data_1d).rolling(window=2).sum().tolist()
    np.testing.assert_array_equal(mapped[1:], series[1:])


def test_diff():
    mapped = send(data_1d, diff())
    np.testing.assert_array_equal(mapped, [2, 1, 4, -5, 1])


def test_wsum_p():
    mapped = send(data_1d, wsum_p(2))
    np.testing.assert_array_equal(mapped, [4, 13, 58, 53, 13])

    mapped2 = send(np.array(data_1d) ** 2, wsum(2))
    np.testing.assert_array_equal(mapped2, [4, 13, 58, 53, 13])


def test_ema():
    mapped = send(data_1d, ema(0.5))
    series = pd.Series(data_1d).ewm(alpha=0.5, adjust=False).mean().tolist()
    np.testing.assert_array_equal(mapped, series)


@pytest.mark.parametrize(
    "indicator_gen, solution", zip(generators_to_test(), solutions)
)
def test_indicators(indicator_gen: Generator, solution):
    mapped = isend(data_1d, indicator_gen)
    np.testing.assert_array_equal(list(mapped), solution)


def test_ma_zero():
    mapped = send([1, 2, 3, 4, 5, 6], ma())
    np.testing.assert_array_equal(mapped, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])


def test_ma5():
    # p_gen = stock_price_generator()
    b_ge = [2.0, 3, 7, 2, 3, 1, 1, 1, 1, 1, 1]
    np.testing.assert_almost_equal(
        send(b_ge, ma(5)), [2.0, 2.5, 4.0, 3.5, 3.4, 3.2, 2.8, 1.6, 1.4, 1, 1]
    )


def test_ma2d(b2d_gen):
    sol = [
        [2, 3],
        [3, 2.5],
        [3, 3],
        [3, 3],
        [2, 3],
        [1.33333333, 2],
        [1, 1.33333333],
        [1, 1],
        [1, 1],
    ]
    np.testing.assert_almost_equal(send(b2d_gen, ma(3)), sol)
