from collections import deque
from typing import Generator, Iterator, TypeVar

import numpy as np
from onstats.stats import wsum as wsum_s, var as var_s, ath as ath_s, ma as ma_s


T = TypeVar("T", np.ndarray, float)

GenStat = Generator[T, T, None]
GenStats = list[GenStat]
TupleGenStat = Generator[T, tuple[T, T], None]


def ath(data_iter: Iterator) -> GenStat:
    """Computes all time high"""
    g = ath_s()
    return (g.send(d) for d in data_iter)


def ma_to_fix(data_iter: Iterator, window: int) -> GenStat:
    """moving average, clean implementation"""
    # something is wrong, but in genreal can be done in this way
    deq = deque()
    value, count = 0, 0
    for value in data_iter:
        deq.append(value)
        if count < window:
            value = (value * count + value) / (count + 1)
            count += 1
        else:
            value = value + (value - deq.popleft()) / window
        yield value


def ma(data_iter: Iterator, window: int) -> GenStat:
    g = ma_s(window)
    return (g.send(d) for d in data_iter)


def wsum(data_iter: Iterator, window: int) -> GenStat:
    """Window sum , if window is zero sum is computed"""
    g = wsum_s(window)
    return (g.send(d) for d in data_iter)


def var(data_iter: Iterator, window: int = 0, ddof: int = 1) -> GenStat:
    """Review if this method is better or worst
    numerically than adding new contribution"""
    g = var_s(window, ddof)
    return (g.send(d) for d in data_iter)


if __name__ == "__main__":
    close = iter((10, 2, 4, 2, 4, 4, 1, 3))
    ma3 = ma(close, 3)
    ma5 = ma(close, 10)

    for i in range(5):
        print(next(ma3) + next(ma5))
