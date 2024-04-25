[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Tests Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](./reports/junit/report.html)

## Generators Based Online Statistics

This package implement online statistics written using python generators, with the only depencency of numpy.

It can callculate basic stats in an online manner with good stability.


## install

> poetry add onstats

> pip install onstats


## How to use:

Import the window / rolling statistic you want to compute and send the values to it:

```python
>>>  from onstats import ma # moving average

>>>  gma = ma(2)  # with window 2
>>>  gma.send(3)
3

>>>  gma.send(5)
4
>>>  gma.send(5)
5
```

If w = 0 the window is infinitelly large , we will compute the normal average.