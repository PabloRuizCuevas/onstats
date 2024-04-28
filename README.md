[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Tests Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](./reports/junit/report.html)

## Generators Based Online Statistics

This package implement online statistics written using python generators, with the only depencency of numpy.

It can calculate basic stats in an online manner with good stability.

The Repo focus in simplicity and compousability, reliying 100% on python Generators.

By desingn window operations increase in window size till the desired size instead of returning Nan in the initialization period as pd.rolling does.

## install

> poetry add onstats

> pip install onstats

## How to use:

Import the window / rolling statistic you want to compute and send the values to it:

```python
>>>  from onstats.stats import ma # moving average

>>>  gma = ma(2)  # with window 2
>>>  gma.send(3)
3

>>>  gma.send(5)
4
>>>  gma.send(5)
5
```

If w = 0 the window is infinitelly large , we will compute the normal average.

You can also feed all the iterator directly:


```python
>>>  from onstats.stats import ma # moving average
>>>  from onstats.util import send

>>>  gma = ma(2)  # with window 2
>>>  send(gma, [3,5,3])
5
```

Or as an iterator

```python
>>>  from onstats.stats import ma # moving average
>>>  from onstats.util import send

>>>  gma = ma(2)  # with window 2
>>>  for d in isend(gma, [3,5,3]):
>>>     print(d)
3
4
5
```

You can also pass 2d np.arrays:

```python
>>>  from onstats.stats import ma # moving average
>>>  from onstats.util import send

>>>  gma = ma(2)  # with window 2
>>>  for d in isend(gma, np.array([[0,0],[1,2],[1,4]])):
>>>     print(d)
[0,0]
[0.5,1]
[1,3]
```


## Supported Stats:

| rolling   | window | infinite | Ddoff | Description                |
|-----------|--------|----------|-------|----------------------------|
| ma        | ✅     | ✅       |       | Moving Average             |
| ema       |        | ✅       |       | Exponential Moving Average |
| var       | ✅     | ✅       | ✅    | Variance                   |
| ath       |        | ✅       |       | All Time High              |
| wsum      | ✅     | ✅       |       | Windowed Sum               |
| cov_xy    | ✅     |          | ✅    | Covariance                 |
| corr_xy   | ✅     |          | ✅    | Correlation                |
| auto_corr | ✅     |          | ✅    | Auto Correlation           |