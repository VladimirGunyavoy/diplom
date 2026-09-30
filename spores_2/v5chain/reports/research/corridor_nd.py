"""Коридор поверх пути адаптивного дерева (очередь research №4), общий вид для SysN: путь replay_value [(слой, τ), …] →
слить подряд одинаковые слои → SLSQP по длительностям: min Σdt при «конец в окне цели» (гладкие неравенства g(x) ≥ 0), dt ≥ 0.
Дерево выбирает топологию (порядок слоёв), оптимизатор — непрерывные моменты переключения. Результат — реальная траектория (верхняя оценка)."""
import numpy as np
from scipy.optimize import minimize


def merge(path):
    out = []
    for s, t in path:
        if out and out[-1][0] == s: out[-1][1] += t
        else: out.append([s, t])
    return [s for s, _ in out], np.array([t for _, t in out], float)


def endpoint(flow, x0, seq, dts):
    x = np.array(x0, float)
    for s, t in zip(seq, dts): x = flow(x, s, t)
    return x


def refine(flow, x0, path, g, tries=3, seed=0):
    """flow(P, s, t); g(x) → массив ≥ 0 внутри окна. Возвращает (T, seq, dts, ok). Старт — длительности пути и их возмущения; лучший допустимый."""
    seq, d0 = merge(path); rng = np.random.default_rng(seed); best = (np.inf, None)
    con = {'type': 'ineq', 'fun': lambda d: np.atleast_1d(g(endpoint(flow, x0, seq, d)))}
    for k in range(tries):
        di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
        r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, None)] * len(d0), constraints=[con],
                     method='SLSQP', options=dict(maxiter=300, ftol=1e-9))
        if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok
