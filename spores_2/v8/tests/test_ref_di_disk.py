"""Эталон ДИ до круга (src/algo/ref_di_disk.py): T_ref ≤ T_point, разность ≲ масштаба цели, сходимость к формуле при RHO→0, сверка с перебором до 2 переключений, сквозная проверка потоком."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from src.algo import ref_di_disk as R


def test_le_point_and_gap():
    X = R.starts(200, 1, .2); T = R.T_ref(X, .2); P = R.T_point(X)
    assert np.isfinite(T).all() and (T <= P + 1e-6).all() and (T >= 0).all()
    assert (P - T).max() < 1.5, (P - T).max()                                  # выигрыш от круга до ~1 (траектория пересекает круг, не доезжая до начала: (.60, −1.25): 1.15 против 2.10); точность — в сверке с перебором


def test_converges_to_point_formula():
    X = R.starts(60, 3, .2); prev = None
    for rho in (.05, .01, .002):
        d = (R.T_point(X) - R.T_ref(X, rho)).max()
        assert prev is None or d < prev + 1e-9; prev = d
    assert prev < 2 * np.sqrt(.002) + 1e-6, prev


def test_vs_brute_two_switch():
    X = R.starts(20, 5, .2); T = R.T_ref(X, .2)
    for x, t in zip(X, T):
        b = R.brute_two_switch(x[0], x[1], .2, step=.05)
        assert t <= b + 1e-6 and b - t < .2, (x, t, b)                            # перебор — верхняя оценка с точностью ~ шаг; не оптимизируем лучше T_ref


def test_inside_and_flow_realises_time():
    assert R.T_ref(np.array([0.05, 0.05]), .2) == 0.
    x0 = np.array([1.3, -.4]); T, info = R.T_ref(x0, .2, return_details=True); s, t1, tot = info
    assert abs(T - tot) < 1e-12
    x, v = x0; dt = 1e-4; t = 0.; u = s
    while np.hypot(x, v) > .2 and t < T + 1e-3:
        if t1 is not None and t >= t1: u = -s
        x, v, t = x + v * dt + u * dt * dt / 2, v + u * dt, t + dt
    assert abs(t - T) < 2e-3, (t, T)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): t0 = time.time(); f(); print('ok', k, '%.1f s' % (time.time() - t0))
