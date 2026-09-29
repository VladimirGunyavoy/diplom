"""Выпрямляющие координаты (АТЛАС §2)."""
A = 1.0


def c_plus(x, v, a=A):  return x - v*v/(2*a)


def c_minus(x, v, a=A): return x + v*v/(2*a)
