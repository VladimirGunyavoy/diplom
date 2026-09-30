"""research hub-research-3: быстрый refine для corridor_nd (прототип, не боевой код). Та же сигнатура, что corridor_nd.refine.
Почему медленно (profile_query.py): SLSQP считает якобиан ограничения конечными разностями → n+1 полных проигрышей пути
ОДНОЙ точкой (Python-rk4 на векторе из 4 чисел, время = накладные numpy), а зазор до препятствий — ns проигрышей сегмента С НАЧАЛА.
Здесь: (1) ∂x_end/∂dt_k = Φ(после k)·f_{s_k}(x_k) — сдвиг конца сегмента k вдоль поля; все n+1 вариантов идут ОДНИМ батчем (строка k+1
«включается» в конце сегмента k), т.е. цена якобиана ≈ цена одного проигрыша; (2) кэш по d (SLSQP зовёт fun и jac в одной точке);
(3) точки зазора — последовательно (p_j = flow(p_{j−1}, t/ns)), их производная по dt_k = (j/ns)·f(p_j) + строки батча.
f_s(x) берётся конечной разностью потока: (flow(x, s, ε) − x)/ε (система не нужна в явном виде)."""
import numpy as np
from scipy.optimize import minimize

EPS = 1e-6
import os; EARLY = bool(os.environ.get('EARLY'))


def _field(flow, X, s, e=1e-5):
    return (flow(X, s, e) - X) / e


def _sweep(flow, x0, seq, d, clear, ns):
    """Батч n+1 строк: строка 0 — номинал, строка k+1 — номинал + EPS·f в конце сегмента k. → (концы (n+1,dim), точки зазора, их якобиан по d)."""
    n = len(seq); X = np.repeat(np.asarray(x0, float)[None], n + 1, 0); P = []; JP = []
    for k, (s, t) in enumerate(zip(seq, d)):
        if clear is None: X = flow(X, s, t)
        else:
            for j in range(1, ns + 1):
                X = flow(X, s, t / ns); p = X[0]; P.append(p)
                J = np.zeros((len(p), n)); J[:, :k] = ((X[1:k + 1] - p) / EPS).T      # по dt_m, m<k — строки батча
                J[:, k] = (j / ns) * _field(flow, p[None], s)[0]                        # по dt_k — сдвиг вдоль поля внутри сегмента
                JP.append(J)
        X[k + 1] = X[0] + EPS * _field(flow, X[:1], s)[0]
    return X, P, JP


def refine_fast(flow, x0, path, g, tries=3, seed=0, clear=None, ns=16):
    from src.atlas6.corridor_nd import merge
    seq, d0 = merge(path); rng = np.random.default_rng(seed); best = (np.inf, None); cache = {}
    def ev(d):
        key = d.tobytes()
        if key in cache: return cache[key]
        X, P, JP = _sweep(flow, x0, seq, d, clear, ns)
        gX = np.array([np.atleast_1d(g(x)) for x in X]); v = gX[0]; J = ((gX[1:] - gX[0]) / EPS).T     # (m, n)
        if clear is not None:
            P = np.array(P); c0 = np.ravel(clear(P)); JPa = np.array(JP)                                  # (np, dim, n)
            dc = np.stack([(np.ravel(clear(P + 1e-6 * JPa[:, :, k])) - c0) / 1e-6 for k in range(len(seq))], -1)
            v = np.concatenate([v, c0]); J = np.vstack([J, dc])
        if len(cache) > 64: cache.clear()
        cache[key] = (v, J); return v, J
    con = {'type': 'ineq', 'fun': lambda d: ev(d)[0], 'jac': lambda d: ev(d)[1]}
    for k in range(tries):
        di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
        r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, 1.5 * d0.sum() + 1)] * len(d0), constraints=[con],
                     method='SLSQP', options=dict(maxiter=300, ftol=1e-9))
        if np.all(ev(r.x)[0] >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok


def refine_seq(flow, x0, path, g, tries=3, seed=0, clear=None, ns=16):
    """Как corridor_nd.refine (якобиан — конечные разности SLSQP), но точки зазора — ПОСЛЕДОВАТЕЛЬНО: p_j = flow(p_{j−1}, t/ns)
    (в исходнике flow(x_сегм, t·j/ns) — каждая точка с начала сегмента, ≈ ns/2 = 8× лишних шагов rk4) + конец пути = последняя точка."""
    from src.atlas6.corridor_nd import merge
    seq, d0 = merge(path); rng = np.random.default_rng(seed); best = (np.inf, None)
    def gg(d):
        x = np.array(x0, float); pts = []
        for sg, t in zip(seq, d):
            if clear is None: x = flow(x, sg, t); continue
            for _ in range(ns): x = flow(x, sg, t / ns); pts.append(x)
        v = np.atleast_1d(g(x))
        return v if clear is None else np.concatenate([v, np.ravel(clear(np.array(pts)))])
    con = {'type': 'ineq', 'fun': gg}
    for k in range(tries):
        di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
        r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, 1.5 * d0.sum() + 1)] * len(d0), constraints=[con],
                     method='SLSQP', options=dict(maxiter=300, ftol=1e-9))
        feas = np.all(con['fun'](r.x) >= -1e-6)
        if feas and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
        if EARLY and k == 0 and not feas: break                 # опыт: недопустимая с первой попытки топология — не повторять
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok
