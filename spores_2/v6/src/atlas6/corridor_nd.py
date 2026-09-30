"""Коридор на адаптивном дереве (nD): дерево даёт топологию (порядок слоёв), SLSQP — длительности (min Σdt, конец в окне).
Кандидаты — стыки прямой спора × ближайшие обратные, в т.ч. с ПРОМАХОМ окна (score = t + промах); top-K разных топологий;
засчитываются только допустимые решения SLSQP. Источник — research hub-research-2 (reports/research/corridor_nd.py, corridor_topk_dd.py)."""
import numpy as np
from scipy.optimize import minimize
from .adaptive_nd import _tree, _fpath, tree_query


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
    """flow(P, s, t); g(x) ≥ 0 внутри окна. → (T, seq, dts, ok); T — лучший ДОПУСТИМЫЙ."""
    seq, d0 = merge(path); rng = np.random.default_rng(seed); best = (np.inf, None)
    con = {'type': 'ineq', 'fun': lambda d: np.atleast_1d(g(endpoint(flow, x0, seq, d)))}
    for k in range(tries):
        di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
        r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, None)] * len(d0), constraints=[con],
                     method='SLSQP', options=dict(maxiter=300, ftol=1e-9))
        if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok


def candidates(S, x0, back, tau, NF, rho, miss, kn=8, dt=None):
    """Стыки → [(score, путь)] по score. miss(X) — расстояние до окна (0 внутри); вошёл: score = t, иначе t + промах."""
    dt = tau / 2 if dt is None else dt
    Pb, Gb, parb, layb, seqs, tree = back; Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True); out = []
    for i in range(len(Pf)):
        fp = _fpath(Pf, parf, layf, i, tau)
        if S.in_goal(Pf[i]): out.append((Gf[i], fp)); continue
        _, j = tree_query(S, tree, Pb, Pf[i], kn)
        for b in j:
            X = np.array(Pf[i]); t = Gf[i]; hit = False; m = np.inf
            for s in seqs[b]:
                for _ in range(2):
                    X = S.wrap(S.flow(X, s, dt)); t += dt
                    if S.in_goal(X): hit = True; break
                    m = min(m, miss(X))
                if hit: break
            out.append((t if hit else t + m, fp + [(s2, tau) for s2 in seqs[b]]))
    return sorted(out, key=lambda c: c[0])


def corridor_query(S, flow, x0, back, tau, NF, rho, g, miss, K=5, kn=8):
    """→ (T, seq, dts) лучший допустимый среди K разных топологий, либо None."""
    seen = set(); best = None
    for _, p in candidates(S, x0, back, tau, NF, rho, miss, kn):
        sq = tuple(merge(p)[0])
        if sq in seen: continue
        seen.add(sq); T, s, d, ok = refine(flow, x0, p, g)
        if ok and (best is None or T < best[0]): best = (T, s, d)
        if len(seen) >= K: break
    return best
