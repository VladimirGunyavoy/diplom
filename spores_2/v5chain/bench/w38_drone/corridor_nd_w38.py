import os
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


def refine(flow, x0, path, g, tries=3, seed=0, clear=None, ns=16, wlim=None):
    """flow(P, s, t); g(x) ≥ 0 внутри окна. → (T, seq, dts, ok); T — лучший ДОПУСТИМЫЙ."""
    seq, d0 = merge(path); rng = np.random.default_rng(seed); best = (np.inf, None)
    import os; CD_ = float(os.environ.get('CLRDT', 0)); NSK = [max(4, int(np.ceil(1.5 * t / CD_))) if CD_ > 0 else ns for t in d0]   # CLRDT: точки зазора каждые ≤ CLRDT/1.5 с, число фиксировано на время SLSQP
    def gg(d):
        if clear is None and wlim is None: return np.atleast_1d(g(endpoint(flow, x0, seq, d)))
        x = np.array(x0, float); pts = []
        for sg, t, nsk in zip(seq, d, NSK):                                  # зазор до препятствий: ns точек на сегмент (фиксированное число — размерность условия не зависит от dt); конец сегмента — последняя точка (один проход)
            for _ in range(nsk): x = flow(x, sg, t / nsk); pts.append(x)      # последовательно: p_j = flow(p_{j-1}, t/ns) (×ns/2 меньше шагов rk4)
        P = np.array(pts); out = [np.atleast_1d(g(x))]
        if clear is not None: out.append(np.ravel(clear(P)))
        if wlim is not None: out.append(np.ravel(wlim[1] - np.abs(P[:, wlim[0]])))      # физический предел |w_i| ≤ WM в ns точках каждого сегмента (слово пользователя 2026-10-01)
        return np.concatenate(out)
    con = {'type': 'ineq', 'fun': gg}
    for k in range(tries):
        di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
        r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, 1.5 * d0.sum() + 1)] * len(d0), constraints=[con],
                     method='SLSQP', options=dict(maxiter=80, ftol=1e-9))
        if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok


NFC = [0]
def candidates(S, x0, back, tau, NF, rho, miss, kn=8, dt=None, hfun=None):
    """Стыки → ([(hit, score, путь)] по (hit, score), число попаданий). Попадание: score = t; промах: t в точке наименьшего промаха + промах.
    Путь обрезан в момент входа в окно / в точке наименьшего промаха (хвост уводит конец с неустойчивого места). Сначала попадания, потом промахи.
    miss(X) — векторный: X (...,d) → расстояние до окна (...,), 0 внутри. Проигрыш всех пар (прямая спора × обратная) — одним батчем по слоям."""
    dt = tau / 2 if dt is None else dt; m = int(round(tau / dt))
    Pb, Gb, parb, layb, seqs, tree = back; Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True, hfun=hfun); out = []; pi_, pb_ = [], []; NFC[0] += len(Pf)
    for i in range(len(Pf)):
        if S.in_goal(Pf[i]): out.append((0, Gf[i], _fpath(Pf, parf, layf, i, tau))); continue
        _, j = tree_query(S, tree, Pb, Pf[i], kn); pi_ += [i] * len(j); pb_ += list(j)
    PT = int(__import__("os").environ.get("PAIRTOP", 0))   # w38: в сшивку идут только PT лучших пар по G0+Gb (латентность)
    if pi_ and PT and len(pi_) > PT:
        _e = np.array([Gf[i] for i in pi_]) + np.array([Gb[j] for j in pb_]); _k = np.argsort(_e, kind="stable")[:PT]; pi_ = [pi_[q] for q in _k]; pb_ = [pb_[q] for q in _k]
    if pi_:
        pi_ = np.array(pi_); pb_ = np.array(pb_); X = np.array([Pf[i] for i in pi_]); G0 = np.array([Gf[i] for i in pi_]); n = len(pi_)
        ln = np.array([len(seqs[b]) for b in pb_]); Sm = np.full((n, max(ln.max(), 1)), -1, int)
        for r, b in enumerate(pb_): Sm[r, :ln[r]] = seqs[b]
        dead = np.zeros(n, bool); hit = np.zeros(n, bool); hit_k = np.zeros(n, int); m0 = miss(X).astype(float); cut_k = np.zeros(n, int); k = 0
        for step in range(Sm.shape[1]):
            for sub in range(m):
                act = ~hit & ~dead & (step < ln)
                if not act.any(): break
                for s_ in range(S.L):
                    sel = act & (Sm[:, step] == s_)
                    if sel.any(): X[sel] = S.wrap(S.flow(X[sel], s_, dt))
                k += 1; dead |= S.blocked(X) & act; act = act & ~dead; ing = S.in_goal(X) & act; hit_k[ing] = k; hit |= ing
                mm = np.where(act & ~ing, miss(X), np.inf); better = mm < m0; m0[better] = mm[better]; cut_k[better] = k
        for r in range(n):
            if dead[r]: continue
            steps = [(Sm[r, q], dt) for q in range(ln[r]) for _ in range(m)]        # по dt-подшагам, как в цикле выше (k считает только активные подшаги)
            fp = _fpath(Pf, parf, layf, pi_[r], tau)
            if hit[r]: kk = hit_k[r]; out.append((0, G0[r] + kk * dt, fp + steps[:kk]))
            else: kk = cut_k[r]; out.append((1, G0[r] + kk * dt + m0[r], fp + steps[:kk]))
    out.sort(key=lambda c: (c[0], c[1])); return out, sum(1 for c in out if c[0] == 0)


_PJ = None; _PJ2 = None
def _cj():   # w37: спекулятивные кандидаты запасного прохода
    S_, x0_, back_, tau_, nf_, rho_, miss_, kn_, hf_ = _PJ2; return candidates(S_, x0_, back_, tau_, nf_, rho_, miss_, kn_, hfun=hf_)
def _pw(a):   # w37: воркер параллельного дожима (fork: состояние запроса в глобале _PJ)
    flow_, x0_, g_, clear_, wlim_, vfy_ = _PJ; p, tr = a
    T, sqr, d, ok = refine(flow_, x0_, p, g_, tries=tr, clear=clear_, wlim=wlim_)
    if ok and vfy_ is not None and not vfy_(sqr, d): ok = False
    return T, sqr, d, ok
def corridor_query(S, flow, x0, back, tau, NF, rho, g, miss, K=5, kn=8, neigh=2, clear=None, tries=3, hfun=None, fs=0, wlim=None, nbest=1, fb=None, vfy=None, kfb=None, _C=None):
    """fb=(NF2, K2): запас на no_path (b20) — если допустимого плана нет, повтор с большим прямым деревом/K (дрон: старт 3 доходит только при NF 6400; 19→20/20); None — выкл. (по умолчанию).
    → (T, seq, dts): лучший допустимый среди K разных топологий-попаданий и K разных топологий-промахов, либо None.
    wlim=(слайс/индексы скоростей, WM): жёсткий предел |w| вдоль пути.
    fs>0: ещё прямые деревья с принудительным первым сегментом каждого слоя (NF=fs на ветку) — разные ветви раскачки/2π (hub-worker-9)."""
    C, nh = _C if _C is not None else candidates(S, x0, back, tau, NF, rho, miss, kn, hfun=hfun); best = None; oks = []
    if fs:
        Ch, Cm = C[:nh], C[nh:]
        for a in range(S.L):
            x1 = S.wrap(S.flow(np.array(x0, float), a, tau))
            if S.blocked(x1) or not S.ok(x1): continue
            Ca, na = candidates(S, x1, back, tau, fs, rho, miss, kn, hfun=hfun); pre = [(a, tau)]
            Ch += [(h, sc + tau, pre + p) for h, sc, p in Ca[:na]]; Cm += [(h, sc + tau, pre + p) for h, sc, p in Ca[na:]]
        Ch.sort(key=lambda c: c[1]); Cm.sort(key=lambda c: c[1]); C = Ch + Cm; nh = len(Ch)
    global _PJ
    NPAR = int(os.environ.get('PARQ', 0)); pool = None; ar = None; spec = None
    if NPAR > 1:
        import multiprocessing as _mp; _PJ = (flow, x0, g, clear, wlim, vfy); global _PJ2
        if fb is not None and int(os.environ.get('SPECFB', 0)): _PJ2 = (S, x0, back, tau, fb[0], rho, miss, kn, hfun)
        pool = _mp.get_context('fork').Pool(NPAR); ar = pool.apply_async(_cj) if (fb is not None and int(os.environ.get('SPECFB', 0))) else None
    def _many(paths, tr):
        if pool is not None: return pool.map(_pw, [(p, tr) for p in paths], chunksize=1)
        out = []
        for p in paths:
            T, sqr, d, ok = refine(flow, x0, p, g, tries=tr, clear=clear, wlim=wlim)
            if ok and vfy is not None and not vfy(sqr, d): ok = False
            out.append((T, sqr, d, ok))
        return out
    jobs = []
    for part in (C[:nh], C[nh:]):
        seen = set()
        for _, _, p in part:
            if not p: continue
            sq = tuple(merge(p)[0])
            if sq in seen: continue
            seen.add(sq); jobs.append(p)
            if len(seen) >= K: break
    for T, sqr, d, ok in _many(jobs, tries):
        if ok: oks.append((T, sqr, d))
        if ok and (best is None or T < best[0]): best = (T, sqr, d)
    oks.sort(key=lambda z: z[0]); starts = [best] + [z for z in oks[:nbest] if z[1] != best[1]][:nbest - 1] if best else []   # nbest>1: соседний поиск и от следующих лучших топологий (m3gu, w28)
    for b0 in starts:
      cur = b0
      for _ in range(neigh):
        sq0, d0 = list(cur[1]), list(cur[2])
        nb = [([a] + sq0, [0.3] + d0) for a in range(S.L) if a != sq0[0]] + [(sq0 + [a], d0 + [0.3]) for a in range(S.L) if a != sq0[-1]]
        nb += [(sq0[1:], d0[1:])] if len(sq0) > 1 else []
        for T, sqr, d, ok in _many([list(zip(sqn, dn)) for sqn, dn in nb], 1):
            if ok and T < cur[0]: cur = (T, sqr, d)
        if cur[0] < best[0]: best = cur
    if pool is not None:
        if best is None and ar is not None: spec = ar.get()
        pool.terminate(); pool.join()
    if best is None and kfb is not None and kfb > K:   # w37: ленивый K — те же кандидаты, больше топологий, только если на малом K плана нет
        return corridor_query(S, flow, x0, back, tau, NF, rho, g, miss, K=kfb, kn=kn, neigh=neigh, clear=clear, tries=tries, hfun=hfun, fs=fs, wlim=wlim, nbest=nbest, fb=fb, vfy=vfy, kfb=None, _C=(C, nh))
    if best is None and fb is not None:
        return corridor_query(S, flow, x0, back, tau, fb[0], rho, g, miss, K=fb[1], kn=kn, neigh=neigh, clear=clear, tries=tries, hfun=hfun, fs=fs, wlim=wlim, nbest=nbest, vfy=vfy, kfb=kfb, _C=spec)
    return best


_JOB = None
def _run(i): S, flow, X, back, tau, NF, rho, g, miss, kw = _JOB; return corridor_query(S, flow, X[i], back, tau, NF, rho, g, miss, **kw)


def corridor_batch(S, flow, X, back, tau, NF, rho, g, miss, procs=16, **kw):
    """Серия запросов параллельно (fork: back строится до вызова и наследуется). → список результатов corridor_query."""
    import multiprocessing as mp
    global _JOB; _JOB = (S, flow, X, back, tau, NF, rho, g, miss, kw)
    with mp.get_context('fork').Pool(min(procs, len(X))) as p: return p.map(_run, range(len(X)), chunksize=1)
