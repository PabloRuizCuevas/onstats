import pytest
import stats.send_operators as gns


@pytest.fixture
def gen_a():
    return iter(range(10))


@pytest.fixture
def gen_b():
    return iter(range(10))


@gns.consumer
def basic_send_generator():
    n = 10
    multiplier = 1
    while True:
        multiplier = yield n * multiplier


def test_repeat_s(gen_a):
    gen_r = gns.repeat_s(gen_a)
    for _ in range(10):
        assert next(gen_r) == 0
    gen_r.send("next")
    for _ in range(10):
        assert next(gen_r) == 1
    gen_r.send(True)
    for _ in range(10):
        assert next(gen_r) == 2
