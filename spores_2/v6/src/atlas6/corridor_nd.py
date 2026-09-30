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
        r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, 1.5 * d0.sum() + 1)] * len(d0), constraints=[con],
                     method='SLSQP', options=dict(maxiter=300, ftol=1e-9))
        if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok


def candidates(S, x0, back, tau, NF, rho, miss, kn=8, dt=None):
    """Стыки → ([(hit, score, путь)] по (hit, score), число попаданий). Попадание: score = t; промах: t в точке наименьшего промаха + промах.
    Путь обрезан в момент входа в окно / в точке наименьшего промаха (хвост уводит конец с неустойчивого места). Сначала попадания, потом промахи."""
    dt = tau / 2 if dt is None else dt
    Pb, Gb, parb, layb, seqs, tree = back; Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True); out = []
    for i in range(len(Pf)):
        fp = _fpath(Pf, parf, layf, i, tau)
        if S.in_goal(Pf[i]): out.append((0, Gf[i], fp)); continue
        _, j = tree_query(S, tree, Pb, Pf[i], kn)
        for b in j:
            X = np.array(Pf[i]); t = Gf[i]; hit = False; m0 = miss(X); tm = t; done = []; cut = []
            for s in seqs[b]:
                for _ in range(2):
                    X = S.wrap(S.flow(X, s, dt)); t += dt; done.append((s, dt))
                    if S.in_goal(X): hit = True; break
                    m = miss(X)
                    if m < m0: m0 = m; tm = t; cut = list(done)
                if hit: break
            out.append((0, t, fp + done) if hit else (1, tm + m0, fp + cut))
    out.sort(key=lambda c: (c[0], c[1])); return out, sum(1 for c in out if c[0] == 0)


def corridor_query(S, flow, x0, back, tau, NF, rho, g, miss, K=5, kn=8, neigh=2):
    """→ (T, seq, dts): лучший допустимый среди K разных топологий-попаданий и K разных топологий-промахов, либо None."""
    C, nh = candidates(S, x0, back, tau, NF, rho, miss, kn); best = None
    for part in (C[:nh], C[nh:]):
        seen = set()
        for _, _, p in part:
            if not p: continue
            sq = tuple(merge(p)[0])
            if sq in seen: continue
            seen.add(sq); T, sqr, d, ok = refine(flow, x0, p, g, tries=3)
            if ok and (best is None or T < best[0]): best = (T, sqr, d)
            if len(seen) >= K: break
    for _ in range(neigh if best else 0):        # локальный поиск по топологиям: соседи лучшей (сегмент другого слоя в начале/конце dt0=.3, без первого), SLSQP каждой
        sq0, d0 = list(best[1]), list(best[2])
        nb = [([a] + sq0, [0.3] + d0) for a in range(S.L) if a != sq0[0]] + [(sq0 + [a], d0 + [0.3]) for a in range(S.L) if a != sq0[-1]]
        nb += [(sq0[1:], d0[1:])] if len(sq0) > 1 else []
        for sqn, dn in nb:
            T, sqr, d, ok = refine(flow, x0, list(zip(sqn, dn)), g, tries=1)
            if ok and T < best[0]: best = (T, sqr, d)
    return best
