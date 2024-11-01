from collections import deque
from typing import Any, Callable, Deque, Generator, TypeVar

import numpy as np

from .util import consumer

T = TypeVar("T", np.ndarray, float)

GenStat = Generator[T, T, None]
GenStats = list[GenStat]
TupleGenStat = Generator[T, tuple[T, T], None]


@consumer
def raw(price) -> GenStat:
    """Dont do anything, but usefull for operating"""
    last = yield 0
    while True:
        last = yield last


@consumer
def ath() -> GenStat:
    """Computes all time high"""
    value = -np.inf
    last = yield 0
    while True:
        value = np.maximum(last, value)
        last = yield value


@consumer
def ma(window: int = 0) -> GenStat:
    """Rolling moving average
    if window = 0 then the total average is computed
    """
    if window == 0:
        yield from avg_()

    deq: Deque = deque()
    value = 0.0
    for count in range(window):
        deq.append((yield value))
        value = (value * count + deq[-1]) / (count + 1)
    del count  # pyright: ignore
    while True:
        deq.append((yield value))
        value = value + (deq[-1] - deq.popleft()) / window


@consumer
def avg() -> GenStat:
    """full average"""
    yield from avg_()


def avg_() -> GenStat:
    """Review if this method is better or worst
    numerically than adding new contribution"""
    count = 0
    cache = yield 0
    while True:
        count += 1
        cache = cache + (yield cache / count)


def isum() -> GenStat:
    """full Sum"""
    last = yield 0
    while True:
        last = last + (yield last)


@consumer
def wsum(window: int = 0) -> GenStat:
    """Window sum , if window is zero sum is computed"""
    if window == 0:
        yield from isum()
    deq: Deque = deque()
    value = 0.0
    for count in range(window):
        deq.append((yield value))
        value = value + deq[-1]
    del count  # pyright: ignore
    while True:
        deq.append((yield value))
        value = value + deq[-1] - deq.popleft()


@consumer
def w_max(window: int) -> GenStat:
    """Computes the max in the last n datapoints"""
    deq: Deque = deque()
    curr_max_ = -np.Inf
    for count in range(window):
        deq.append((yield curr_max_))
        curr_max_ = max(curr_max_, deq[-1])
    while True:
        deq.append((yield curr_max_))
        out = deq.popleft()
        """Maybe is more efficient using sortedlist there is the package."""
        curr_max_ = max(deq) if out == curr_max_ else max(curr_max_, deq[-1])


@consumer
def w_min(window: int) -> GenStat:
    """Computes the max in the last n datapoints"""
    deq: Deque = deque()
    curr_min_ = np.Inf
    for count in range(window):
        deq.append((yield curr_min_))
        curr_min_ = max(curr_min_, deq[-1])
    while True:
        deq.append((yield curr_min_))
        out = deq.popleft()
        """Maybe is more efficient using sortedlist there is the package."""
        curr_max_ = max(deq) if out == curr_min_ else max(curr_min_, deq[-1])


@consumer
def sum_f(window: int, function: Callable) -> GenStat:
    """Same as sum but applying a function to the values sent"""
    deq: Deque = deque()
    value = 0.0
    for _ in range(window):
        deq.append((yield value))
        value = value + function(deq[-1])
    while True:
        deq.append((yield value))
        value = value + function(deq[-1]) - function(deq.popleft())


@consumer
def wsum_p(window: int, pow: int = 2) -> GenStat:
    """Sum powers of the number"""
    deq: Deque = deque()
    value = 0.0
    for _ in range(window):
        deq.append((yield value))
        value = value + deq[-1] ** pow
    while True:
        deq.append((yield value))
        value = value + deq[-1] ** pow - deq.popleft() ** pow


def var_(ddof: int = 1) -> GenStat:
    """Computes variance"""
    mav_g, sum_g, sump_g = ma(), wsum(), wsum()
    window = 0
    last = yield 0
    while True:
        window += 1
        mav, suma, sump2 = mav_g.send(last), sum_g.send(last), sump_g.send(last**2)
        last = yield (sump2 - 2 * mav * suma + window * mav**2) / max(window - ddof, 1)


@consumer
def var(window: int = 0, ddof: int = 1) -> GenStat:
    """m**2 + 1/n sum_i( x_i**2 - 2 mu x_i )
    1/2 sum_i( x_i**2 - 2 mu x_i )
    """
    if window == 0:
        yield from var_(ddof)

    mav_g, sum_g, sump_g = ma(window), wsum(window), wsum(window)
    last = yield 0
    for count in range(1, window):
        mav, suma, sump2 = mav_g.send(last), sum_g.send(last), sump_g.send(last**2)
        last = yield (sump2 - 2 * mav * suma + count * mav**2) / max(count - ddof, 1)
    while True:
        mav, suma, sump2 = mav_g.send(last), sum_g.send(last), sump_g.send(last**2)
        last = yield (sump2 - 2 * mav * suma + window * mav**2) / (window - ddof)


@consumer
def cov_xy(window: int, ddof=1) -> TupleGenStat:
    """1/n sum_i((x_i-m_x) * (x_j m_y) )
    x_i*x_j - x_i*m_y - x_j*m_i  + m_x * m_y
    In the case x=y then it becomes same thing as var
    """
    x, y = yield 0
    m_x, m_y = ma(window), ma(window)
    s_x, s_y, s_xy = wsum(window), wsum(window), wsum(window)
    for count in range(1, window):
        mux, muy, sx, sy, sxy = m_x.send(x), m_y.send(y), s_x.send(x), s_y.send(y), s_xy.send(x * y)
        x, y = yield (sxy - muy * sx - mux * sy + count * muy * mux) / max(count - ddof, 1)
    while True:
        mux, muy, sx, sy, sxy = m_x.send(x), m_y.send(y), s_x.send(x), s_y.send(y), s_xy.send(x * y)
        x, y = yield (sxy - muy * sx - mux * sy + window * muy * mux) / (window - ddof)


@consumer
def corr_xy(window: int, ddof: int = 0) -> TupleGenStat:
    """Correlation"""
    cov = cov_xy(window, ddof)
    var_x, var_y = var(window, ddof), var(window, ddof)
    x, y = yield 0.0
    while True:
        cov_val = cov.send((x, y))
        x_var, y_var = var_x.send(x), var_y.send(y)
        x, y = yield cov_val / np.sqrt(x_var * y_var)


@consumer
def auto_corr(window: int, time: int = 10, ddof: int = 0) -> GenStat:
    """Auto correlation"""
    corr = corr_xy(window, ddof)
    x_delay = delay(time)
    x = yield 0.0
    while True:
        x = yield corr.send((x, x_delay.send(x)))


@consumer
def delay(periods: int, default: Any = 0) -> GenStat:
    """Delays the iterator"""
    deq: Deque = deque()
    deq.append((yield default))
    for _ in range(periods):
        deq.append((yield default))
    while True:
        deq.append((yield deq.popleft()))


@consumer
def ema(alpha: float | None = None, com: float | None = None, halflife: float | None = None) -> GenStat:
    """
    TODO implement as well adjusted version based on https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
    """
    if alpha is None:
        if com is not None:
            alpha = 2 / (com + 1)
        elif halflife is not None:
            alpha = 1 - 0.5 ** (1 / halflife)
    assert alpha is not None, "Either alpha or com or halflife must be provided"
    assert 0 < alpha < 1, "Alpha must be between 0 and 1"

    last = yield 0
    current_ema = last
    while True:
        current_ema = last * alpha + current_ema * (1 - alpha)
        last = yield current_ema


@consumer
def diff() -> GenStat:
    """the difference of two iterators"""
    old = 0.0
    last = yield 0.0
    while True:
        new = yield last - old
        old = last
        last = new


@consumer
def percentual_diff() -> GenStat:
    """if open close available is the same as percentual_spread"""
    old = 1
    last = yield 0
    while True:
        new = yield (last - old) / old
        old = last
        last = new


@consumer
def percentual_spread() -> TupleGenStat:
    """relative difference between high and low"""
    low, high = yield 0.0
    while True:
        low, high = yield (high - low) / low


@consumer
def normalize(window: int = 0, sample_freq: int = 10) -> GenStat:
    """Online normalization"""
    count = 0
    avg_g, var_g = ma(window), var(window)
    last = yield 0
    s_var = 1
    for _ in range(2):
        m, v = avg_g.send(last), var_g.send(last)
        last = yield (last - m)
    while True:
        if count % sample_freq == 0:
            m, v = avg_g.send(last), var_g.send(last)
            s_var = np.sqrt(v)
        last = yield (last - m) / s_var  # pyright: ignore
        count += 1


# TA


@consumer
def k_stoch(window: int = 14) -> GenStat:
    """Stoch window is typically in days"""
    last = yield 0
    l14_g, h14_g = w_min(window), w_max(window)
    while True:
        l14 = l14_g.send(last)
        k = (last - l14) / (h14_g.send(last) - l14) * 100
        yield k


@consumer
def d_stoch(ma_window: int = 3, k_window: int = 14) -> GenStat:
    """Stoch window is typically in days"""
    k_gen = k_stoch(k_window)
    ma_g = ma(ma_window)
    last = yield 0
    while True:
        last = yield ma_g.send(k_gen.send(last))
