import asyncio
import operator
import uuid
from typing import Any, AsyncIterator, Callable, Generator, Iterator, TypeVar

import numpy as np

from onstats.stats import (
    percentual_diff,
    var as var_s,
    wsum as wsum_s,
    normalize as normalize_s,
    ma as ma_s,
    delay as delay_s,
    corr_xy as corr_xy_s,
    ath as ath_s,
    ema as ema_s,
)

G = TypeVar("G")
H = TypeVar("H")


async def sequential_azip(gen1: AsyncIterator[G], gen2: AsyncIterator[H]) -> AsyncIterator[tuple[G | None, H | None]]:
    """The bad thing is that as gen1, gen2 can share asycronous deps,
    a task group raises runtime errors"""
    while True:
        # try:
        #     async with asyncio.TaskGroup() as tg:
        #         task1 = tg.create_task(anext(gen1))
        #         task2 = tg.create_task(anext(gen2))
        #     yield task1.result(), task2.result()
        yield await anext(gen1), await anext(gen2)


class Lgen:
    """Locked Generators,
    when next(ge) anext(ge) is executed it calls the next value on the generator and locks
    the value for future next calls till unlock is called.

    This class is intended for working both for async and sync generators
    """

    registry = {}

    # could specify the lock by nesting giving additional argument
    def __init__(self, it: AsyncIterator | Iterator):
        self.iterator = it  # sinc or async
        self.cache = None
        self.uuid = uuid.uuid4()  # slow

    def __next__(self):
        if self.uuid not in Lgen.registry:
            Lgen.registry[self.uuid] = next(self.iterator)
        return Lgen.registry[self.uuid]

    def __iter__(self):
        return self

    async def __anext__(self):
        if self.uuid not in Lgen.registry:
            Lgen.registry[self.uuid] = await anext(self.iterator)
        return Lgen.registry[self.uuid]

    def __aiter__(self):
        return self

    def __getitem__(self, key: int) -> 'Lgen':
        return delay(self, key)

    @classmethod
    def unlock(cls):
        """time passes calling unlock"""
        cls.registry = {}

    @classmethod
    def consume(cls, iterators: list["Lgen"]) -> Generator[list[Any], None, None]:
        """Pass iterators to consume with time passing"""
        while True:
            cls.unlock()
            try:
                yield [next(it) for it in iterators]
            except StopIteration:
                break

    @classmethod
    async def a_consume(cls, iterators: list["Lgen"]):
        """Pass iterators to consume with time passing"""
        while True:
            cls.unlock()
            try:
                yield [await anext(it) for it in iterators]
            except StopIteration:
                break

    @classmethod
    def ga(cls, func: Callable[[Any], Callable[[Any], "Lgen"]]):
        """Use as decorator for convert iterators into Ge"""

        def wrapper(*args, **kw):
            gen = Lgen(func(*args, **kw))
            return gen

        wrapper.__name__ = func.__name__
        wrapper.__dict__ = func.__dict__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def nclass(self, other, op):
        """Creates Ge class from two Ge classes

        for now it supports either all sync or all async, but could support also mixed
        """

        if isinstance(self.iterator, Iterator):
            if isinstance(other, (int, float)):
                return self.__class__(op(x, other) for x in self)
            else:
                return self.__class__(op(x, y) for x, y in zip(self, other))

        elif isinstance(self.iterator, AsyncIterator):
            if isinstance(other, (int, float)):
                return self.__class__(op(x, other) async for x in self)
            else:
                return self.__class__(op(x, y) async for x, y in sequential_azip(self, other))

    def __add__(self, other: Iterator | float):
        return self.nclass(other, operator.add)

    def __radd__(self, other: Iterator | float):
        return self.__add__(other)

    def __sub__(self, other: 'Lgen | Iterator | float'):
        return self.nclass(other, operator.sub)

    def __mul__(self, other: Iterator):
        return self.nclass(other, operator.mul)

    def __rmul__(self, other: Iterator):
        return self.__mul__(other)

    def __truediv__(self, other: Iterator | float):
        return self.nclass(other, operator.truediv)

    def __floordiv__(self, other: Iterator):
        return self.nclass(other, operator.floordiv)

    def __mod__(self, other: Iterator):
        return self.nclass(other, operator.mod)

    def __pow__(self, other: Iterator):
        return self.nclass(other, operator.pow)

    def __and__(self, other: Iterator):
        return self.nclass(other, operator.and_)

    def __eq__(self, other: 'Lgen | Iterator | float'):
        return self.nclass(other, operator.eq)

    def __lt__(self, other: 'Lgen | Iterator | float'):
        return self.nclass(other, operator.lt)

    def __le__(self, other: 'Lgen | Iterator | float'):
        return self.nclass(other, operator.le)

    def __gt__(self, other: 'Lgen | Iterator | float'):
        return self.nclass(other, operator.gt)

    def __ge__(self, other: 'Lgen | Iterator | float'):
        return self.nclass(other, operator.ge)


def bfor(g: Generator, data_iter: Lgen) -> Lgen:
    """Returns a sync or async iterator given a Locked Generator"""
    if isinstance(data_iter.iterator, AsyncIterator):
        return (g.send(d) async for d in data_iter)
    elif isinstance(data_iter.iterator, Iterator):
        return (g.send(d) for d in data_iter)
    raise

@Lgen.ga
def sma(data_iter: Lgen, window: int) -> Lgen:
    """moving average of n window lenght"""
    return bfor(ma_s(window), data_iter)


@Lgen.ga
def vwma(source: Lgen, volume: Lgen, window: int) -> Lgen:
    """moving average of n window lenght"""
    return sma(source * volume, window) / sma(volume, window)


@Lgen.ga
def ema(data_iter: Lgen, alpha: float | None = None, com: float | None = None, halflife: float | None = None) -> Lgen:
    """moving average of n window lenght"""
    return bfor(ema_s(alpha, com, halflife), data_iter)


@Lgen.ga
def ath(data_iter: Lgen) -> Lgen:
    """All time high"""
    return bfor(ath_s(), data_iter)


@Lgen.ga
def wsum(data_iter: Lgen) -> Lgen:
    """window sum"""
    return bfor(wsum_s(), data_iter)


@Lgen.ga
def var(data_iter: Lgen, window: int, ddof: int = 1) -> Lgen:
    """variance"""
    return bfor(var_s(window, ddof), data_iter)


@Lgen.ga
def diff(data_iter: Lgen) -> Lgen:
    """variance"""
    return data_iter - data_iter[1]


@Lgen.ga
def rdiff(data_iter: Lgen) -> Lgen:
    """variance"""
    # a = data_iter   (a - data_iter[1])/a
    return bfor(percentual_diff(), data_iter)


@Lgen.ga
def delay(data_iter: Lgen, periods: int, default: float = 0):
    """Delays returning the value of the generator by n periods,
    returns default while teh periods are reached"""
    return bfor(delay_s(periods, default), data_iter)


@Lgen.ga
def normalize(data_iter: Lgen, window: int = 0, sample_freq: int = 10):
    """Normalizes de data online substracting the average and dividing by
    the rolling std"""
    return bfor(normalize_s(window, sample_freq), data_iter)


@Lgen.ga
def corr_xy(data_iter_a: Lgen, data_iter_b: Lgen, window: int, ddof: int = 0):
    """correlation"""
    return bfor(corr_xy_s(window, ddof), Lgen(i for i in zip(data_iter_a, data_iter_b)))


@Lgen.ga
def auto_corr(data_iter: Lgen, window: int, ddof: int = 0):
    """auto correlation"""
    return bfor(corr_xy_s(window, ddof), data_iter)


def crossover(a: Lgen, b: Lgen) -> Lgen:
    """Check if a crossed over b"""
    return (a <= b)[1] & (a > b)


def crossunder(a: Lgen, b: Lgen) -> Lgen:
    """Check if a crossed under b"""
    return (a >= b)[1] & (a < b)


def test_data_gen(price: float = 100, mu: float = 0.0002, sigma: float = 0.005) -> Iterator[float]:
    """A generator for synthetic stock prices using Geometric Brownian Motion."""
    while True:
        random_shock = np.random.normal(loc=0, scale=sigma)
        daily_return = np.exp((mu - 0.5 * sigma**2) + random_shock)
        price *= daily_return
        yield price


# Nothing important from here on


async def a_test_data_gen(price: float = 100, mu: float = 0.0002, sigma: float = 0.005) -> AsyncIterator[float]:
    """test"""
    for v in test_data_gen(price=price, mu=mu, sigma=sigma):
        await asyncio.sleep(0.1)
        yield v


async def amain():
    price = Lgen(i async for i in a_test_data_gen())
    comb = sma(price, 2) + sma(price, 3)
    async for i in Lgen.a_consume([comb]):
        print(i)


def main():
    price = Lgen(i for i in test_data_gen())
    comb = sma(price, 2) + sma(price, 3)
    for i in Lgen.consume([comb]):
        print(i)


if __name__ == "__main__":
    # main()
    asyncio.run(amain())
