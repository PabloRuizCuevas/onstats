from typing import Any, Callable, Generator, Iterable, Iterator, Literal, NoReturn, TypeAlias
from pytrade.util.operators import gnext


Repeat: TypeAlias = Generator[Any, Literal[False] | None, Any]


def repeat_s(gen: Iterator) -> Repeat:
    """Repeats the last value of a generator. till a signal to proceed is received."""
    # repeat = None
    for val in gen:
        repeat = None
        while repeat is None:
            repeat = yield val
        # repeat = None


def run(repeators: list[Repeat], *args: Iterator) -> Generator:
    """Couples two generators. to a common dependency. generator,
    This may feel a bit spooky action at a distance,
    as the send method is used to control the flow of the generators,
    but the dependency of them it can't be seen from this function"""
    try:
        while True:
            yield gnext(*args)
            msend(False, *repeators)
    except StopIteration:
        return None


def consumer(func):
    """avoid priming the generator"""

    def wrapper(*args, **kw):
        gen = func(*args, **kw)
        next(gen)
        return gen

    wrapper.__name__ = func.__name__
    wrapper.__dict__ = func.__dict__
    wrapper.__doc__ = func.__doc__
    return wrapper


### Send operators ###
# 1 data 1 gen  -> gen.send
# n data 1 gen  -> isend  returns Generator
# n data 1 gen  -> send   list(isend)
# 1 data n gen  -> msend
# n data n gen  -> msendg


def isend(data: Iterable, gen: Generator) -> Generator[Generator, None, None]:
    """equivalent of map(gen.send, data)"""
    return (gen.send(d) for d in data)


def send(data: Iterable, gen: Generator) -> list[Generator]:
    # list(isend(data, gen))
    return [gen.send(d) for d in data]


def msend(data: Any, *generators: Generator) -> list[Generator]:
    return [gen.send(data) for gen in generators]


def msendg(data: Iterator, *generators: Generator) -> Generator[list[Generator], None, None]:
    return (msend(d, *generators) for d in data)


@consumer
def smap(func: Callable, gen: Generator) -> Generator:
    """Expects primed generator or consumer pattern one,
    applys a function and preserves the data sent to the generator"""
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield func(gen.send(val))


@consumer
def wrap(gen: Generator, wrap: Callable) -> Generator:
    """Wraps data sent to generator with a function"""
    s = yield None
    while True:
        s = yield gen.send(wrap(s))


@consumer
def compose(final_gen: Generator, preprocess_gen: Generator) -> Generator:
    """Expects primed generator or consumer pattern one"""
    val = yield None
    while True:
        val = yield final_gen.send(preprocess_gen.send(val))


@consumer
def promise() -> Generator[Any | None, Any, NoReturn]:
    val = None
    while True:
        ret = yield val
        val = ret if ret is not None else val


def catched_gnext(generators: Iterator[Generator]):
    """catches value of generators based on id, not recommended"""
    results = {}
    for g in generators:
        if id(g) not in results:
            results[id(g)] = next(g)
        yield results[id(g)]


# Operators


@consumer
def add(gen_a: Generator, gen_b: Generator) -> Generator:
    # Prime the generators to get them to the first yield
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield gen_a.send(val) + gen_b.send(val)


@consumer
def add_msend(gen_a: Generator, gen_b: Generator) -> Generator:
    # on doing this
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield gen_a.send(val) + gen_b.send(val)


@consumer
def sub(gen_a: Generator, gen_b: Generator) -> Generator:
    # Prime the generators to get them to the first yield
    # a, b = next(gen_a), next(gen_b)
    val: Any = yield None
    while True:
        # Yield the sum of current values
        val = yield gen_a.send(val) - gen_b.send(val)


@consumer
def div(gen_a: Generator, gen_b: Generator) -> Generator:
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield gen_a.send(val) / gen_b.send(val)


@consumer
def percentual_diff(gen_a: Generator, gen_b: Generator) -> Generator:
    """div(sub(gen_a, gen_b), gen_b)"""
    val = yield None
    while True:
        a, b = gen_a.send(val), gen_b.send(val)
        val = yield (a - b) / b


def apply_two(op: Callable, gen_a: Generator, gen_b: Generator) -> Generator:
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield op(gen_a.send(val), gen_b.send(val))


def apply(op: Callable, *generators: Generator[Any, Any, None]) -> Generator:
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield op(*[gen.send(val) for gen in generators])
