from collections import deque
from typing import Generator, Iterator, TypeVar

import numpy as np
from onstats.stats import wsum as wsum_s, var as var_s, ath as ath_s
from onstats.util import consumer, timed, TIME

T = TypeVar("T", np.ndarray, float)

GenStat = Generator[T, T, None]
GenStats = list[GenStat]
TupleGenStat = Generator[T, tuple[T, T], None]



@consumer
def next_on_send(generator: Iterator[T]) -> Iterator[T]:
    """ Retains the data till a signal is recieved """
    for val in generator:
        while True:
            go_next = yield val
            if go_next:
                break

def ath(data_iter) -> GenStat:
    """Computes all time high"""
    return (ath_s().send(d) for d in data_iter)


def ma(data_iter: Iterator, window: int) -> GenStat:
    """moving average, clean implementation"""
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

tma = timed(ma)

def wsum(data_iter: Iterator, window: int) -> GenStat:
    """Window sum , if window is zero sum is computed"""
    return (wsum_s(window).send(d) for d in data_iter)


def var(data_iter: Iterator, window: int = 0, ddof: int = 1) -> GenStat:
    """Review if this method is better or worst
    numerically than adding new contribution"""
    return (var_s(window, ddof).send(d) for d in data_iter)



if __name__ == "__main__":
    import operator
    from types import GeneratorType
    from fishhook import hook

    def g_op(gen1, gen2, op) -> Generator:
        v = yield
        while True:
            v = yield op(gen1.send(v), gen2.send(v))

    @hook(GeneratorType)
    def __add__(self, other) -> GenStat:
        g = g_op(self, other, operator.add)
        g.send(None)
        return g

    close_dat = (2,3,1,4,3,6,5,4,5,4,3,6,7,5,6,7,3,3,2)
    close = next_on_send(close_dat)
    ma3, ma10 = ma(close, 3), ma(close, 10)
    ma_comp = ma3 + ma10

    @timed
    def tcloseg():
        yield from close_dat

    tclose = tcloseg()

    ma_test = tma(tclose, 3) + tma(tclose, 10)
    #ma_comp = (a+b for a,b in zip(ma3,ma10))
    #limitation of this approach is that ma

    for i in range(10):
        TIME +=1
        print("old",close.send(True), next(ma_comp))
        print("new",next(ma_test))
