from typing import Any, Callable, Generator, Iterable, Iterator, TypeVar

T = TypeVar("T")


def consumer(func):
    """avoid priming the generator
    Also defined in more_itertools/more.py
    """

    def wrapper(*args, **kw):
        gen = func(*args, **kw)
        next(gen)
        return gen

    wrapper.__name__ = func.__name__
    wrapper.__dict__ = func.__dict__
    wrapper.__doc__ = func.__doc__
    return wrapper


@consumer
def compose(final_gen: Generator, preprocess_gen: Generator) -> Generator:
    """Expects primed generator or consumer pattern one"""
    val = yield None
    while True:
        val = yield final_gen.send(preprocess_gen.send(val))


def gmap(op: Callable, *generators: Generator[Any, Any, None]) -> Generator:
    """apply operation to gennerators"""
    val = yield None
    while True:
        # Yield the sum of current values
        val = yield op(*[gen.send(val) for gen in generators])


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
