from onstats.stats import wsum as wsum_s, var as var_s, ath as ath_s, ma as ma_s
from typing import AsyncIterator, Generator, Iterator, TypeVar, AsyncGenerator
import numpy as np


T = TypeVar("T", np.ndarray, float)

GenStat = AsyncGenerator[T, T, None]
GenStats = list[GenStat]
TupleGenStat = Generator[T, tuple[T, T], None]


async def ath(data_iter: AsyncIterator) -> GenStat:
    """Computes all time high"""
    g = ath_s()
    return (g.send(d) async for d in data_iter)


def ma(data_iter: AsyncIterator, window: int) -> GenStat:
    """ma"""
    g = ma_s(window)
    return (g.send(d) async for d in data_iter)


# two ways or async function or normal function and return async iterator,
# the crazy fact is that the one with comprehension can be done programatically
async def wsum(data_iter: AsyncIterator, window: int) -> GenStat:
    """Window sum , if window is zero sum is computed"""
    g = wsum_s(window)
    async for d in data_iter:
        return g.send(d)


async def var(data_iter: AsyncIterator, window: int = 0, ddof: int = 1) -> GenStat:
    """Review if this method is better or worst
    numerically than adding new contribution"""
    g = var_s(window, ddof)
    return (g.send(d) async for d in data_iter)


if __name__ == "__main__":
    close = iter((10, 2, 4, 2, 4, 4, 1, 3))
    ma3 = ma(close, 3)
    ma5 = ma(close, 10)

    for i in range(5):
        print(next(ma3) + next(ma5))
