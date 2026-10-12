"""C2 дифдрайв-ромб 3D: v7 grow3 (w21: RMIN .04 MINROWS 5 NFAIL 60 PAR 1 VF 5 NA 4 VFR 1) + агент с финишем стрельбой. Реплика rollout() с записью (u,dt)-сегментов (оригинал не меняем). Запуск (aida, из bench/): python3 spores_dd.py"""
import os, sys, time, json
for k, v in dict(RMIN='.04', MINROWS='5', NFAIL='60', PAR='1', VF='5', NA='4', VFR='1', TQDM_MI='1000', OMP_NUM_THREADS='1').items(): os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, itertools, common
import grow3 as G

def shoot2(y, tmax):
    """как G.shoot, но возвращает ещё длительности дуг"""
    from scipy.optimize import minimize
    best = (np.inf, None, None); R = G.RHO * .9
    for tp in G.TOPS:
        def end(d):
            z = np.array(y, float)
            for k, dt in zip(tp, d): z = G.arc(z, G.US[k], dt)
            return np.r_[z[:2], G.wrap(z[2])]
        cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
        for d0 in (np.full(len(tp), tmax / (2 * len(tp))), np.full(len(tp), tmax / len(tp))):
            r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * len(tp), constraints=cons, options=dict(maxiter=100, ftol=1e-9))
            if r.success and np.all(np.abs(end(r.x)) <= G.RHO) and r.x.sum() < best[0] and (not G.DISCS or G._free(y, tp, r.x)): best = (r.x.sum(), tp, r.x.copy())
    return best

def rollout_rec(A, Q, tmax=40.):
    vf = G.VF; Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); vbest = np.full(n, np.inf); lr = np.zeros(n, bool); vis = [dict() for _ in range(n)]; nopr = np.zeros(n, int); fin = np.zeros(n, bool)
    seg = [[] for _ in range(n)]; US = G.US; DTN = G.DTN
    def fin_add(i, tf, tp, dd): T[i] += tf; done[i] = True; fin[i] = True; seg[i] += [(US[k], float(d)) for k, d in zip(tp, dd)]
    for _ in range(int(tmax / DTN)):
        if done.all(): break
        v = A.vstar(Y)
        for i in np.flatnonzero(~done & (v <= vf)):
            tf, tp, dd = shoot2(Y[i], 2.5 * max(v[i], .2))
            if tp is not None and (v[i] <= G.VF0 or tf <= G.VFR * v[i] + .05): fin_add(i, tf, tp, dd)
        if done.all(): break
        tgs = np.stack([A.tgoal(Y, u) for u in US], 1); Ys = [G.step(Y, u) for u in US]; Jv = DTN + np.stack([A.vstar(y_) for y_ in Ys], 1)
        if G.LRTA:
            for i in np.flatnonzero(lr & ~done):
                for ki in range(len(US)): Jv[i, ki] += G.LRTA * vis[i].get(G._key(Ys[ki][i]), 0)
                vis[i][G._key(Y[i])] = vis[i].get(G._key(Y[i]), 0) + 1
        J = np.minimum(tgs, Jv); k = J.argmin(1); tg = tgs[np.arange(n), k]; stuck = J.min(1) >= G.BIG / 2
        vc = A.vstar(Y); nopr = np.where(vc <= vbest - G.NOPD, 0, nopr + 1); vbest = np.minimum(vbest, vc)
        for i in np.flatnonzero(~done & (nopr >= G.NOPR)):
            nopr[i] = 0; lr[i] = True; tf, tp, dd = shoot2(Y[i], 2.5 * max(float(vc[i]) if vc[i] < G.BIG / 2 else 4., .2))
            if tp is not None: fin_add(i, tf, tp, dd)
        for i in np.flatnonzero(~done & stuck):
            vv = float(A.vstar(Y[i:i + 1])[0]); tf, tp, dd = shoot2(Y[i], 2.5 * max(vv if vv < G.BIG / 2 else 4., .2))
            if tp is not None: fin_add(i, tf, tp, dd)
        stuck &= ~done; act = ~done & ~stuck; Yn = Y.copy()
        for ki, ui in enumerate(US):
            m = act & (k == ki)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        for i in np.flatnonzero(act): seg[i].append((US[k[i]], float(tg[i]) if np.isfinite(tg[i]) else DTN))
        Y = G.wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= G.ingoal(Y) | stuck
    T[~(G.ingoal(Y) | fin)] = np.inf; return T, seg

if __name__ == '__main__':
    e = common.make('dd'); t0 = time.perf_counter(); A = G.Atlas(); t_grow = time.perf_counter() - t0; print('built', [len(l) for l in A.layers], round(t_grow), flush=True)
    t0 = time.perf_counter(); A.solve(); t_solve = time.perf_counter() - t0; print('solved', round(t_solve), flush=True)
    Q = e.X0; res = []; Ts = []; tq = []
    for i in range(len(Q)):
        t1 = time.perf_counter(); T, seg = rollout_rec(A, Q[i:i + 1]); dtq = time.perf_counter() - t1; tq.append(dtq); sg = seg[0]
        Tv, why = common.verify(e, Q[i], [list(u) for u, d in sg], [d for u, d in sg]); Ts.append(Tv); res.append(dict(i=i, T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_query=dtq, nseg=len(sg))); print(i, res[-1], flush=True)
    t1 = time.perf_counter(); rollout_rec(A, Q); t_batch = time.perf_counter() - t1
    s = common.summarize(e, Ts); s.update(env='dd', method='spores v7 grow3 (VF 5 NA 4 VFR 1)', t_grow=t_grow, t_solve=t_solve, t_build=t_grow + t_solve, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)), t_batch20=t_batch,
                                      cells=[len(l) for l in A.layers], nodes=int(A.N), hw='CPU (PAR=1: 4 процесса на рост; solve 1 proc)', per_start=res, Tref=e.Tref.tolist())
    json.dump(s, open(os.path.join(common.DATA, 'dd_spores.json'), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
