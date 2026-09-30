"""research hub-research-3: flow4 для ОДНОЙ точки на чистом Python (math, без numpy) — прототип. Батч (ndim>1) → исходный flow4.
Причина: в refine поток зовётся точкой (вектор из 4 чисел), время = накладные numpy (~25 ufunc/stack/ones_like на вызов f4), не арифметика.
Проверка и замер: python3 flow4_scalar.py (из spores_2/v6)."""
import math
import numpy as np


def flow4s(x, s, t, dt_max=0.02, A=2.5, B=0.5, D=1.0, LAYERS4=((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
    x = np.asarray(x, float)
    if x.ndim > 1:
        from src.atlas6.manip2dyn import flow4
        return flow4(x, s, t, dt_max)
    ta, tb = LAYERS4[s]; q1, q2, w1, w2 = (float(v) for v in x); n = max(1, int(math.ceil(abs(t) / dt_max))); h = t / n
    def f(q2, w1, w2):
        c = math.cos(q2); hh = -B * math.sin(q2); r1 = ta + hh * (2 * w1 * w2 + w2 * w2); r2 = tb - hh * w1 * w1
        m11 = A + 2 * B * c; m12 = D + B * c; det = m11 * D - m12 * m12
        return (D * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det
    for _ in range(n):
        a1, b1 = f(q2, w1, w2); k1 = (w1, w2, a1, b1)
        a2, b2 = f(q2 + h / 2 * k1[1], w1 + h / 2 * a1, w2 + h / 2 * b1); k2 = (w1 + h / 2 * a1, w2 + h / 2 * b1, a2, b2)
        a3, b3 = f(q2 + h / 2 * k2[1], w1 + h / 2 * a2, w2 + h / 2 * b2); k3 = (w1 + h / 2 * a2, w2 + h / 2 * b2, a3, b3)
        a4, b4 = f(q2 + h * k3[1], w1 + h * a3, w2 + h * b3); k4 = (w1 + h * a3, w2 + h * b3, a4, b4)
        q1 += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); q2 += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        w1 += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); w2 += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return np.array([q1, q2, w1, w2])


if __name__ == '__main__':
    import sys, time, platform; sys.path.insert(0, '.')
    from src.atlas6.manip2dyn import flow4
    rng = np.random.default_rng(0); X = rng.uniform(-2, 2, (200, 4)); err = 0
    for x in X:
        for s in range(4): err = max(err, np.abs(flow4s(x, s, 0.7, 0.05) - flow4(x, s, 0.7, 0.05)).max())
    for f in (flow4, flow4s):
        t = time.perf_counter()
        for x in X: f(x, 1, 0.4, 0.05)
        print(platform.node(), f.__name__, 'мкс/вызов (t=.4, 8 шагов rk4): %.0f' % ((time.perf_counter() - t) / len(X) * 1e6))
    print('max|разница|', err)
