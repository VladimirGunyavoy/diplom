"""research-17 (PLAN 24): эталон 4D двойного интегратора в коробку |x|,|v| <= RHO. Состояние (x1, x2, v1, v2), ẍi = ui, |ui| <= 1, оси независимы.
Множество времён попадания оси в коробку — [Ti*, ∞) (лишнее время — дугами +−+), ⇒ T* = max(T1*, T2*). Ti* — min по ≤ 3 дугам ±1 (замкнутая форма, SLSQP, мультистарт),
min с точечной формулой DI (верхняя оценка). Старты: rng(1), 60, [-2, 2]^4. Выход: di4_ref_60.npy — (60, 7): x1 x2 v1 v2 T1 T2 T*."""
import numpy as np, itertools, sys
from scipy.optimize import minimize
RHO = float(sys.argv[1]) if len(sys.argv) > 1 else .05
def arc(x, v, u, t): return x + v * t + u * t * t / 2, v + u * t
def tpoint(x, v):                                                       # точный T* DI в точку (0, 0), |u| <= 1
    s = x + v * abs(v) / 2; u = -np.sign(s) if s != 0 else -np.sign(v)
    if u == 0: return 0.
    return -u * v + 2 * np.sqrt(v * v / 2 - u * x)
def tbox(x, v):
    if abs(x) <= RHO and abs(v) <= RHO: return 0.
    best = tpoint(x, v)
    for n in (1, 2, 3):
        for s0 in (1., -1.):
            us = [s0 * (-1) ** k for k in range(n)]
            def end(d):
                z = (x, v)
                for u, t in zip(us, d): z = arc(*z, u, max(t, 0.))
                return np.array(z)
            for d0 in (np.full(n, best / n if np.isfinite(best) else 1.), np.full(n, best / (2 * n) if np.isfinite(best) else .5)):
                r = minimize(lambda d: d.sum(), d0, method='SLSQP', bounds=[(0, None)] * n, constraints=[{'type': 'ineq', 'fun': lambda d: .999 * RHO - np.abs(end(d))}], options=dict(maxiter=200, ftol=1e-12))
                if r.success and np.all(np.abs(end(r.x)) <= RHO + 1e-9) and r.x.sum() < best: best = r.x.sum()
    return best
rng = np.random.default_rng(1); Q = rng.uniform(-2, 2, (60, 4))
T1 = np.array([tbox(q[0], q[2]) for q in Q]); T2 = np.array([tbox(q[1], q[3]) for q in Q]); Ts = np.maximum(T1, T2)
P = np.array([max(tpoint(q[0], q[2]), tpoint(q[1], q[3])) for q in Q])
np.save('di4_ref_60.npy', np.c_[Q, T1, T2, Ts]); print('T* box: mean %.3f min %.3f max %.3f; box/point mean %.4f min %.4f' % (Ts.mean(), Ts.min(), Ts.max(), (Ts / P).mean(), (Ts / P).min()))
