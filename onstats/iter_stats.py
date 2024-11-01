from typing import AsyncIterator, Generator, Iterator, TypeVar
import numpy as np
from onstats.stats import wsum as wsum_s, var as var_s, ath as ath_s, ma as ma_s


T = TypeVar("T", np.ndarray, float)

GenStat = Generator[T, T, None]
GenStats = list[GenStat]
TupleGenStat = Generator[T, tuple[T, T], None]


def create_iterator(g: Generator, data_iter: Generator | AsyncIterator):
    if isinstance(data_iter, AsyncIterator):
        return (g.send(d) async for d in data_iter)
    elif isinstance(data_iter, Iterator):
        return (g.send(d) for d in data_iter)


def ma(data_iter: Generator, window: int) -> Generator:
    return create_iterator(ma_s(window), data_iter)


def ath(data_iter: Generator) -> Generator:
    return create_iterator(ath_s(), data_iter)


def wsum(data_iter: Generator) -> Generator:
    return create_iterator(wsum_s(), data_iter)


def var(data_iter: Generator, window: int, ddof: int) -> Generator:
    return create_iterator(var_s(window, ddof), data_iter)


if __name__ == "__main__":
    close = iter((10, 2, 4, 2, 4, 4, 1, 3))
    ma3 = ma(close, 3)
    ma5 = ma(close, 10)

    for i in range(5):
        print(next(ma3) + next(ma5))
