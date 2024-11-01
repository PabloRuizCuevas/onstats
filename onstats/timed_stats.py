from typing import Callable, Generator, Iterator, TypeVar, Any
from onstats.iter_stats import ma as ma_i, var as var_i, ath as ath_i, wsum as wsum_i
import numpy as np
import uuid


T = TypeVar("T", np.ndarray, float)

GenStat = Generator[T, T, None]
GenStats = list[GenStat]
TupleGenStat = Generator[T, tuple[T, T], None]


class Ge:
    """Lock Constrained Generators,
    When next(ge) is executed it calls the next value on the generator
    when is executed a second time the previous value is returned
    when
    """

    registry = {}

    # could specify the lock by nesting giving additional argument
    def __init__(self, it: Iterator) -> Iterator:
        self.iterator = it
        self.cache = None
        self.uuid = uuid.uuid4()  # slow

    def __next__(self):
        if self.uuid not in Ge.registry:
            Ge.registry[self.uuid] = next(self.iterator)
        return Ge.registry[self.uuid]

    def __iter__(self):
        return self

    def forward(self):
        """next value after passing time"""
        Ge.unlock()
        return next(self)

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
    def ga(cls, func: Callable[[Any], Callable[[Any], "Ge"]]):
        """Use as decorator for convert iterators into Ge"""

        def wrapper(*args, **kw):
            gen = Ge(func(*args, **kw))
            return gen

        wrapper.__name__ = func.__name__
        wrapper.__dict__ = func.__dict__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def __add__(self, other: Iterator):
        return self.__class__(x + y for x, y in zip(self, other))

    def __sub__(self, other: Iterator):
        return self.__class__(x - y for x, y in zip(self, other))

    def __mul__(self, other: Iterator):
        return self.__class__(x * y for x, y in zip(self, other))

    def __truediv__(self, other: Iterator):
        return self.__class__(x / y for x, y in zip(self, other))

    def __floordiv__(self, other: Iterator):
        return self.__class__(x // y for x, y in zip(self, other))

    def __mod__(self, other: Iterator):
        return self.__class__(x % y for x, y in zip(self, other))

    def __pow__(self, other: Iterator):
        return self.__class__(x**y for x, y in zip(self, other))

    def __eq__(self, other: Iterator):
        return self.__class__(x == y for x, y in zip(self, other))

    def __lt__(self, other: Iterator):
        return self.__class__(x < y for x, y in zip(self, other))

    def __le__(self, other: Iterator):
        return self.__class__(x <= y for x, y in zip(self, other))

    def __gt__(self, other: Iterator):
        return self.__class__(x > y for x, y in zip(self, other))

    def __ge__(self, other: Iterator):
        return self.__class__(x >= y for x, y in zip(self, other))


ma = Ge.ga(ma_i)
ath = Ge.ga(ath_i)
var = Ge.ga(var_i)
wsum = Ge.ga(wsum_i)


def test_data_gen(price: float = 100, mu: float = 0.0002, sigma: float = 0.005) -> Iterator[float]:
    """A generator for synthetic stock prices using Geometric Brownian Motion."""
    while True:
        random_shock = np.random.normal(loc=0, scale=sigma)
        daily_return = np.exp((mu - 0.5 * sigma**2) + random_shock)
        price *= daily_return
        yield price


if __name__ == "__main__":
    from itertools import islice
    from collections import deque
    from uniplot import plot_gen

    price = Ge(i for i in (i for i in test_data_gen()))
    signals = Ge.consume((price, ma(price, 20), ma(price, 40), ath(price), ma(price, 10) > ma(price, 30)))

    y = deque([i for i in islice(signals, 150)])
    g = plot_gen(lines=True, color=["blue", "red", "green"])
    next(g)

    for i in range(1000):
        y.append(next(signals))
        y.popleft()
        t = np.array(y).T
        x = list(range(len(t[0])))
        g.send(((t[0], t[1], t[2]), (x, x, x), {}))
