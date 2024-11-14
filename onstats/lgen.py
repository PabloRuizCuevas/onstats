import asyncio
import operator
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Generator, Iterator, TypeVar
from __future__ import annotations
import numpy as np

from onstats.stats import ath as ath_s
from onstats.stats import corr_xy as corr_xy_s
from onstats.stats import delay as delay_s
from onstats.stats import ema as ema_s
from onstats.stats import ma as ma_s
from onstats.stats import normalize as normalize_s
from onstats.stats import percentual_diff
from onstats.stats import var as var_s
from onstats.stats import wsum as wsum_s

type UnkIterator[T] = AsyncIterator[T] | Iterator[T]
type LgenOperable = "Lgen | Iterator | float"

async def sequential_azip[G,H](gen1: AsyncIterator[G], 
                               gen2: AsyncIterator[H]) -> AsyncIterator[tuple[G, H]]:
    """The bad thing is that as gen1, gen2 can share asycronous deps,
    a task group raises runtime errors"""
    while True:
        # try:
        #     async with asyncio.TaskGroup() as tg:
        #         task1 = tg.create_task(anext(gen1))
        #         task2 = tg.create_task(anext(gen2))
        #     yield task1.result(), task2.result()
        yield await anext(gen1), await anext(gen2)


class Lgen[T]:
    """Locked Generators,
    when next(ge) anext(ge) is executed it calls the next value on the generator and locks
    the value for future next calls till unlock is called.

    This class is intended for working both for async and sync generators
    """

    registry: dict[uuid.UUID, 'Lgen'] = {}

    # could specify the lock by nesting giving additional argument
    def __init__(self, it: UnkIterator[T]):
        self.iterator: UnkIterator[T] = it  # sinc or async
        self.cache = None
        self.uuid = uuid.uuid4()  # slow

    def __next__(self) -> T:
        if self.uuid not in Lgen.registry:
            Lgen.registry[self.uuid] = next(self.iterator)
        return Lgen.registry[self.uuid]

    def __iter__(self):
        return self

    async def __anext__(self) -> T:
        if self.uuid not in Lgen.registry:
            Lgen.registry[self.uuid] = await anext(self.iterator)
        return Lgen.registry[self.uuid]

    def __aiter__(self):
        return self

    def __getitem__(self, key: int) -> "Lgen":
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
    async def a_consume(cls, iterators: list[Lgen]) -> AsyncGenerator[list[Lgen], None, None]:
        """Pass iterators to consume with time passing"""
        while True:
            cls.unlock()
            try:
                yield [await anext(it) for it in iterators]
            except StopIteration:
                break

    def nclass(self, other: LgenOperable, op: Callable[[T, T], T]) -> Lgen[T]:
        """Creates Ge class from two Ge classes

        for now it supports either all sync or all async, but could support also mixed
        """
        gen = None
        if isinstance(self.iterator, Iterator):
            if isinstance(other, (int, float)):
                gen = (op(x, other) for x in self)
            else:
                gen = (op(x, y) for x, y in zip(self, other))

        elif isinstance(self.iterator, AsyncIterator):
            if isinstance(other, (int, float)):
                gen = (op(x, other) async for x in self)
            else:
                gen = (op(x, y) async for x, y in sequential_azip(self, other))
        return self.__class__(gen)

    def __add__(self, other: LgenOperable) -> Lgen[T]:
        return self.nclass(other, operator.add)

    def __radd__(self, other: LgenOperable) -> Lgen[T]:
        return self.__add__(other)

    def __sub__(self, other: LgenOperable) -> Lgen[T]:
        return self.nclass(other, operator.sub)

    def __mul__(self, other: LgenOperable) -> Lgen[T]:
        return self.nclass(other, operator.mul)

    def __rmul__(self, other: LgenOperable) -> Lgen[T]:
        return self.__mul__(other)

    def __truediv__(self, other: LgenOperable) -> Lgen[T]:
        return self.nclass(other, operator.truediv)

    def __floordiv__(self, other: LgenOperable) -> Lgen[T]:
        return self.nclass(other, operator.floordiv)

    def __mod__(self, other: LgenOperable) -> Lgen[T]:
        return self.nclass(other, operator.mod)

    def __pow__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.pow)

    def __and__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.and_)

    def __eq__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.eq)

    def __lt__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.lt)

    def __le__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.le)

    def __gt__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.gt)

    def __ge__(self, other: LgenOperable)-> Lgen[T]:
        return self.nclass(other, operator.ge)

    @classmethod
    def from_s_gen[T](cls, s_gen: Generator[T, Any, None], data_iter: Lgen[T] | UnkIterator[T]) -> Lgen[T]:
        """Returns a sync or async iterator given a Locked Generator"""
        iterator = getattr(data_iter, "iterator", data_iter)
        match iterator:
            case it if isinstance(it, AsyncIterator):
                return Lgen(s_gen.send(d) async for d in data_iter)
            case it if isinstance(it, Iterator):
                return Lgen(s_gen.send(d) for d in data_iter)
            case _:
                raise TypeError("data_iter.iterator must be an Iterator or AsyncIterator")


def sma(data_iter: Lgen, window: int) -> Lgen:
    """moving average of n window lenght"""
    return Lgen.from_s_gen(ma_s(window), data_iter)


def vwma(source: Lgen, volume: Lgen, window: int) -> Lgen:
    """moving average of n window lenght"""
    return sma(source * volume, window) / sma(volume, window)


def ema(data_iter: Lgen, com: float | None = None, alpha: float | None = None, halflife: float | None = None) -> Lgen:
    """moving average of n window lenght"""
    return Lgen.from_s_gen(ema_s(alpha, com, halflife), data_iter)


def ath(data_iter: Lgen) -> Lgen:
    """All time high"""
    return Lgen.from_s_gen(ath_s(), data_iter)


def wsum(data_iter: Lgen) -> Lgen:
    """window sum"""
    return Lgen.from_s_gen(wsum_s(), data_iter)


def var(data_iter: Lgen, window: int, ddof: int = 1) -> Lgen:
    """variance"""
    return Lgen.from_s_gen(var_s(window, ddof), data_iter)


def diff(data_iter: Lgen) -> Lgen:
    """variance"""
    return data_iter - data_iter[1]


def rdiff(data_iter: Lgen) -> Lgen:
    """variance"""
    # a = data_iter   (a - data_iter[1])/a
    return Lgen.from_s_gen(percentual_diff(), data_iter)


def delay(data_iter: Lgen, periods: int, default: float = 0) -> Lgen:
    """Delays returning the value of the generator by n periods,
    returns default while teh periods are reached"""
    return Lgen.from_s_gen(delay_s(periods, default), data_iter)


def normalize(data_iter: Lgen, window: int = 0, sample_freq: int = 10) -> Lgen:
    """Normalizes de data online substracting the average and dividing by
    the rolling std"""
    return Lgen.from_s_gen(normalize_s(window, sample_freq), data_iter)


def corr_xy(data_iter_a: Lgen, data_iter_b: Lgen, window: int, ddof: int = 0) -> Lgen:
    """correlation"""
    return Lgen.from_s_gen(corr_xy_s(window, ddof), Lgen(i for i in zip(data_iter_a, data_iter_b)))


def auto_corr(data_iter: Lgen, window: int, ddof: int = 0) -> Lgen:
    """auto correlation"""
    return Lgen.from_s_gen(corr_xy_s(window, ddof), data_iter)


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
