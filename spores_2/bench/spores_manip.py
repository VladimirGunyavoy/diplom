"""C2 манипулятор 4D: growN (снимок wb5: слои g300d3 GS 300 DELTA .3 MAXC 3000 PESS 1, solve SONE=1 GPU, финиш shoot_pol FINGRID 2 VF 2) — запросы по 20 стартам OCP.
MODE=load (по умолч.): атлас с V из ~/spore_v5/wb5/m45/A_g300so.pkl — только запросы; MODE=full: слои (CPU) → solve (GPU, venv-sb3) → запросы, с таймингом (см. run_manip_full.sh).
Реплика rollout() с записью (u,dt) сегментов. Запуск (aida, из c1/spores_2/bench): python3 spores_manip.py"""
import os, sys, time, json, pickle
ENV = dict(SYS='manip', KF='11', M='3', FRAC='.5', OVH='.5', MINROWS='3', GLIM='0', RMAX='.6', TMAX='3', SBCAUS='1', SBLAY='0', NOLATCH='1', FINGRID='2', VF='2', PESS='1', GPUMAX='1000000', GS='300', DELTA='.3', MAXC='3000', SONE='1', TQDM_MI='1000', OMP_NUM_THREADS='2')
for k, v in ENV.items(): os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.expanduser('~/spore_v5/c1/manip_src'); sys.path.insert(0, HERE); sys.path.insert(0, SRC)
import numpy as np, common
import growN as G, finish_gen as FG
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]

def shoot_pol_rec(y, tmax, NA, inits=2, H=.1):
    from scipy.optimize import minimize
    T0, tp = FG.shoot_grid(y, G.f, G.US, G.RHOV, G.wrapy, tmax, NA, H=H)
    if tp is None: return T0, tp, None
    R = .9 * np.asarray(G.RHOV); best = (T0, tp, None); n = len(tp)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FG.flow(z, G.US[k], max(dt, 0.), G.f)
        return G.wrapy(z)
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if r.success and np.all(np.abs(end(r.x)) <= np.asarray(G.RHOV)) and r.x.sum() < best[0]: best = (float(r.x.sum()), tp, r.x.copy())
    if best[2] is None: return best[0], tp, None                       # сетка без доводки: длительности = шаг сетки (нет в API) → отметим
    return best

def rollout_rec(A, Q, tmax=40.):
    E = os.environ.get; DTN = G.DTN; US = G.US; VF = G.VF
    Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); tried = np.zeros(n, bool); fin = np.zeros(n, bool); seg = [[] for _ in range(n)]; nodur = np.zeros(n, bool)
    for _ in range(int(tmax / DTN)):
        if done.all(): break
        tgs = np.stack([A.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([A.vstar(G.step(Y, u)) for u in US], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
        for i in np.flatnonzero(~done & ~tried & (J.min(1) <= VF)):
            tried[i] = True; Ts, tp, dd = shoot_pol_rec(Y[i], G.FTMAX, int(E('NA', 3)))
            if np.isfinite(Ts):
                T[i] += Ts; done[i] = True; fin[i] = True
                if dd is None: nodur[i] = True
                else: seg[i] += [(US[kk], float(d)) for kk, d in zip(tp, dd)]
        stuck = J.min(1) >= G.BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
        for ki, ui in enumerate(US):
            m = act & (k == ki)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        for i in np.flatnonzero(act): seg[i].append((US[k[i]], float(tg[i]) if np.isfinite(tg[i]) else DTN))
        Y = G.wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= G.ingoal(Y) | stuck
    T[~G.ingoal(Y) & ~fin] = np.inf; return T, seg, nodur

if __name__ == '__main__':
    e = common.make('manip'); t_grow = t_solve = None; mode = os.environ.get('MODE', 'load')
    if mode == 'load':
        d = pickle.load(open(os.path.expanduser('~/spore_v5/wb5/m45/A_g300so.pkl'), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; A.n_it = 0; A.edges = 0
    Q = e.X0; res = []; Ts = []; tq = []
    for i in range(len(Q)):
        t1 = time.perf_counter(); T, seg, nodur = rollout_rec(A, Q[i:i + 1]); dtq = time.perf_counter() - t1; tq.append(dtq); sg = seg[0]
        Tv, why = common.verify(e, Q[i], [list(u) for u, d in sg], [d for u, d in sg]); Ts.append(Tv)
        res.append(dict(i=i, T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_query=dtq, nseg=len(sg), finish_no_durations=bool(nodur[0]))); print(i, res[-1], flush=True)
    s = common.summarize(e, Ts); s.update(env='manip', method='spores growN g300 SONE VF2 PESS1 4x3000 (%s)' % mode, t_grow=t_grow, t_solve=t_solve, t_build=(t_grow or 0) + (t_solve or 0) if t_grow else None, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)),
                                      cells=[len(l) for l in A.layers], nodes=int(A.N), hw='CPU (запросы)', per_start=res, Tref=e.Tref.tolist())
    s['weight'] = common.deep_weight(A)
    json.dump(s, open(os.path.join(common.DATA, 'manip_spores%s.json' % ('' if mode == 'full' else '_load')), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
