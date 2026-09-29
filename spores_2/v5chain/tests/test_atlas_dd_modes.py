"""to_chart∘from_chart = id и совпадение с численным интегрированием кинематики. Запуск: python3 tests/test_atlas_dd_modes.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas_dd import Straight, Rotate, Arc

V, W = 1.0, 1.5
MODES = [(Straight(s, V), (s * V, 0.0)) for s in (1, -1)] + [(Rotate(s, W), (0.0, s * W)) for s in (1, -1)] \
      + [(Arc(sv, sw, V, W), (sv * V, sw * W)) for sv in (1, -1) for sw in (1, -1)]


def integrate(st, u, T, n=20000):
    x, y, th = st; v, w = u; dt = T / n
    for _ in range(n):   # средняя точка — 2-й порядок
        thm = th + 0.5 * w * dt
        x += v * np.cos(thm) * dt; y += v * np.sin(thm) * dt; th += w * dt
    return np.array([x, y, th])


def test_modes():
    rng = np.random.default_rng(1)
    for m, u in MODES:
        for _ in range(20):
            st = (*rng.uniform(-2, 2, 2), rng.uniform(-3, 3))
            label, s = m.to_chart(st)
            assert np.allclose(m.from_chart(label, s), st, atol=1e-12)
            T = rng.uniform(0.1, 2.0)
            end = integrate(st, u, T)
            s_end = s + m.rate * T
            assert np.allclose(m.from_chart(label, s_end), end, atol=1e-6), (type(m).__name__, u)
            assert np.allclose(m.to_chart(m.from_chart(label, s_end))[0], label, atol=1e-9)   # label сохраняется вдоль траектории


if __name__ == "__main__":
    test_modes(); print("OK")
