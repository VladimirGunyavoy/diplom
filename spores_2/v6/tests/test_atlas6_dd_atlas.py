"""Атлас дифдрайва (ромб-U): на осях V совпадает с точным значением с поправкой на допуск цели (Rθ)."""
import sys; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.dd_atlas import solve_dd

for h in (0.5, 0.25):
    r = solve_dd(h=h); V, xs, ths, hth = r['V'], r['xs'], r['ths'], r['hth']
    j = np.argmin(abs(xs)); i = np.argmin(abs(xs - 2)); k0 = np.argmin(abs(ths)); kp = np.argmin(abs(ths - np.pi / 2))
    assert abs(V[i, j, k0] - 2.0) < 1e-6, V[i, j, k0]                              # задом по прямой
    assert abs(V[j, j, kp] - (np.pi / 2 - hth)) < 1e-6, V[j, j, kp]               # поворот на месте до допуска цели
    assert abs(V[j, i, kp] - (2 + np.pi / 2 - hth)) < 1e-6, V[j, i, kp]           # задом + поворот
    assert V.max() < 2 * np.pi + 5, V.max()
    print('h', h, 'iters', r['iters'], 'OK')

from src.atlas6.dd_atlas import rollout, V_at
A = solve_dd(h=0.25); rng = np.random.default_rng(0); ok = 0; ratios = []
for _ in range(20):
    p0 = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]); path, t, done = rollout(A, p0)
    ok += done; ratios.append(t / max(V_at(A, p0), 1e-9))
print('rollout дошёл', ok, '/20; время/V: mean %.3f max %.3f' % (np.mean(ratios), np.max(ratios)))
assert ok >= 18

from src.atlas6.dd_atlas import corridor
rng = np.random.default_rng(1); r_ = []
for _ in range(10):
    p0 = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]); c, T, res, T0 = corridor(A, p0)
    assert res < 1e-6, res; r_.append(T / V_at(A, p0))
print('коридор до точки цели: время/V mean %.3f max %.3f' % (np.mean(r_), np.max(r_)))
assert np.max(r_) < 1.2
