from fishhook import hook
from types import GeneratorType
import operator
import itertools

def g_op(gen1, gen2, op):
    v = yield
    while True:
        v = yield op(gen1.send(v), gen2.send(v))
        
@hook(GeneratorType)
def __add__(self, other):
    g = g_op(self,other, operator.add)
    g.send(None)
    return g

@hook(GeneratorType)
def __mul__(self, other):
    g = g_op(self, other, operator.mul)
    g.send(None)
    return g

@hook(GeneratorType)
def __sub__(self, other):
    g = g_op(self, other, operator.sub)
    g.send(None)
    return g

@hook(GeneratorType)
def __pow__(self, other):
    g = g_op(self, other, operator.pow)
    g.send(None)
    return g

def count():
    yield from itertools.count()
    
c1 = count()
c2 = count()
c3 = count()

s = c1 * c3 + c2

next(s)