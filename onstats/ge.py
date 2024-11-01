from typing import AsyncIterator, Callable, Generator, Iterator, TypeVar, Any
import numpy as np
import uuid
import asyncio
import operator
from onstats.stats import wsum as wsum_s, var as var_s, ath as ath_s, ma as ma_s


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


class Ge:
    """Lock Constrained Generators,
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
        if self.uuid not in Ge.registry:
            Ge.registry[self.uuid] = next(self.iterator)
        return Ge.registry[self.uuid]

    def __iter__(self):
        return self

    async def __anext__(self):
        if self.uuid not in Ge.registry:
            Ge.registry[self.uuid] = await anext(self.iterator)
        return Ge.registry[self.uuid]

    def __aiter__(self):
        return self

    @classmethod
    def unlock(cls):
        """time passes calling unlock"""
        cls.registry = {}

    @classmethod
    def consume(cls, iterators: list["Ge"]):
        """Pass iterators to consume with time passing"""
        while True:
            cls.unlock()
            try:
                yield [next(it) for it in iterators]
            except StopIteration:
                break

    @classmethod
    async def a_consume(cls, iterators: list["Ge"]):
        """Pass iterators to consume with time passing"""
        while True:
            cls.unlock()
            try:
                yield [await anext(it) for it in iterators]
            except StopIteration:
                break

    @classmethod
    def ga(cls, func: Callable[[Any], Callable[[Any], "Ge"]]):
        """Use as decorator for convert iterators into Ge"""

        def wrapper(*args, **kw):
            gen = Ge(func(*args, **kw))
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
            return self.__class__(op(x, y) for x, y in zip(self, other))
        elif isinstance(self.iterator, AsyncIterator):
            return self.__class__(op(x, y) async for x, y in sequential_azip(self, other))

    def __add__(self, other: Iterator):
        return self.nclass(other, operator.add)

    def __sub__(self, other: Iterator):
        return self.nclass(other, operator.sub)

    def __mul__(self, other: Iterator):
        return self.nclass(other, operator.mul)

    def __truediv__(self, other: Iterator):
        return self.nclass(other, operator.truediv)

    def __floordiv__(self, other: Iterator):
        return self.nclass(other, operator.floordiv)

    def __mod__(self, other: Iterator):
        return self.nclass(other, operator.mod)

    def __pow__(self, other: Iterator):
        return self.nclass(other, operator.pow)

    def __eq__(self, other: Iterator):
        return self.nclass(other, operator.eq)

    def __lt__(self, other: Iterator):
        return self.nclass(other, operator.lt)

    def __le__(self, other: Iterator):
        return self.nclass(other, operator.le)

    def __gt__(self, other: Iterator):
        return self.nclass(other, operator.gt)

    def __ge__(self, other: Iterator):
        return self.nclass(other, operator.ge)


def bfor(g: Generator, data_iter: Ge):
    if isinstance(data_iter.iterator, AsyncIterator):
        return (g.send(d) async for d in data_iter)
    elif isinstance(data_iter.iterator, Iterator):
        return (g.send(d) for d in data_iter)


@Ge.ga
def ma(data_iter: Ge, window: int) -> Ge:
    return bfor(ma_s(window), data_iter)


@Ge.ga
def ath(data_iter: Ge) -> Ge:
    return bfor(ath_s(), data_iter)


@Ge.ga
def wsum(data_iter: Ge) -> Ge:
    return bfor(wsum_s(), data_iter)


@Ge.ga
def var(data_iter: Ge, window: int, ddof: int) -> Ge:
    return bfor(var_s(window, ddof), data_iter)


def test_data_gen(price: float = 100, mu: float = 0.0002, sigma: float = 0.005) -> Iterator[float]:
    """A generator for synthetic stock prices using Geometric Brownian Motion."""
    while True:
        random_shock = np.random.normal(loc=0, scale=sigma)
        daily_return = np.exp((mu - 0.5 * sigma**2) + random_shock)
        price *= daily_return
        yield price


async def a_test_data_gen(price: float = 100, mu: float = 0.0002, sigma: float = 0.005) -> AsyncIterator[float]:
    for v in test_data_gen(price=price, mu=mu, sigma=sigma):
        await asyncio.sleep(0.1)
        yield v


async def amain():
    price = Ge(i async for i in a_test_data_gen())
    comb = ma(price, 2) + ma(price, 3)
    async for i in Ge.a_consume([comb]):
        print(i)


def main():
    price = Ge(i for i in test_data_gen())
    comb = ma(price, 2) + ma(price, 3)
    for i in Ge.consume([comb]):
        print(i)


if __name__ == "__main__":
    # main()
    asyncio.run(amain())
