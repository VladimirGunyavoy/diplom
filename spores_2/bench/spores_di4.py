"""CA1(1): ДИ 4D, споры — полные слои Lfull (25 650 клеток, 9.46M узлов; b5/r22), V из r22/18 (один стенсил на группу, SBLAY 1, Якоби PESS 1 → V_Lfull.npy), агент + финиш shoot_pol VF 1 с записью (u, dt) для проверяльщика.
Заготовка (цифры r22/18, не пересчитаны здесь): стенсилы 1355 с GPU + solve 66.5 с (GPU Якоби 170 проходов); слои Lfull — времени построения нет в логах r22 (b5) → см. knowledge; вес: слои + V.
python spores_di4.py (aida, cwd ~/spore_v5/c1/di4_run, venv-sb3 не нужен для запросов, но growN импортирует torch → venv-sb3) → data/di4_spores.json"""
import os, sys, time, json, pickle
for k, v in dict(SYS='di4', M='3', SIDE='0', GM='.25', GLIM='0', FRAC='1', RHO='.35', PESS='1', FH='.1').items(): os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.expanduser(os.environ.get('SRC_DIR', '~/spore_v5/c1/di4_run')); sys.path.insert(0, HERE); sys.path.insert(0, SRC)
import numpy as np, common
import growN as G, fin, finish_gen as FGm
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from scipy.optimize import minimize
US = G.US

def shoot_pol_rec(y, tmax, NA=3):
    T0, tp = fin.shoot_grid(y, G.f, US, G.RHOV, G.wrapy, tmax, NA)
    if tp is None: return T0, tp, None
    R = .9 * np.asarray(G.RHOV); n = len(tp); best = (np.inf, tp, None)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FGm.flow(z, US[k], max(dt, 0.), G.f)
        return G.wrapy(z)
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if np.all(np.abs(end(r.x)) <= np.asarray(G.RHOV)) and r.x.sum() < best[0]: best = (float(r.x.sum()), tp, r.x.copy())
    return best if best[2] is not None else (np.inf, tp, None)

def rollout_rec(A, Q, tmax=40.):
    DTN = G.DTN; VF = 1.; Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); tried = np.zeros(n, bool); seg = [[] for _ in range(n)]; fin_ = np.zeros(n, bool)
    for _ in range(int(tmax / DTN)):
        if done.all(): break
        tgs = np.stack([A.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([A.vstar(G.step(Y, u)) for u in US], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
        for i in np.flatnonzero(~done & ~tried & (J.min(1) <= VF)):
            tried[i] = True; Ts, tp, dd = shoot_pol_rec(Y[i], G.FTMAX, int(os.environ.get('NA', 3)))
            if np.isfinite(Ts): T[i] += Ts; done[i] = True; fin_[i] = True; seg[i] += [(US[kk], float(d)) for kk, d in zip(tp, dd)]
        stuck = J.min(1) >= G.BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
        for ki, ui in enumerate(US):
            m = act & (k == ki)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        for i in np.flatnonzero(act): seg[i].append((US[k[i]], float(tg[i]) if np.isfinite(tg[i]) else DTN))
        Y = G.wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= G.ingoal(Y) | stuck
    T[~G.ingoal(Y) & ~fin_] = np.inf; return T, seg

if __name__ == '__main__':
    e = common.make('di4'); AVP = os.environ.get('AVPKL')      # c7: b12 — слои+V из growN DUMP; без него — Lfull r22/18
    if AVP: d_ = pickle.load(open(os.path.expanduser(AVP), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']
    else: d_ = pickle.load(open(os.path.join(os.path.expanduser('~/spore_v5/wb5/p38'), 'Lfull.pkl'), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = np.load(os.path.join(SRC, 'V_Lfull.npy'))
    G.VF = 1.; res = []; Ts = []; tq = []
    for i in range(len(e.X0)):
        t1 = time.perf_counter(); T, seg = rollout_rec(A, e.X0[i:i + 1]); dtq = time.perf_counter() - t1; tq.append(dtq); sg = seg[0]
        Tv, why = common.verify(e, e.X0[i], [np.array(u, float) for u, d in sg], [d for u, d in sg]); Ts.append(Tv)
        res.append(dict(i=i, keep=bool(e.keep[i]), T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_query=dtq, nseg=len(sg))); print(i, res[-1], flush=True)
    s = common.summarize(e, Ts); s.update(env='di4', **(dict(method='споры b12 (COVAFF PARG=6 PARB=8 QFAST SPU JCOMP, 4×1600) + финиш shoot_pol VF1', t_build=float(os.environ['TB']), t_solve=float(os.environ['TS']), t_build_note='замер b12/c7 под нагрузкой aida') if AVP else dict(method='споры Lfull (r22/18, один стенсил/группу) + финиш shoot_pol VF1', t_build=1355. + 66.5, t_build_note='слои Lfull (b5) без времени', t_stencils=1355., t_solve=66.5)), t_query_med=float(np.median(tq)), t_query_sum=float(np.sum(tq)), t_query_p90=float(np.percentile(tq, 90)), cells=[len(l) for l in A.layers], nodes=int(A.N), hw='GPU (стенсилы, solve; цифры r22/18), CPU (запросы)', per_start=res, Tref=e.Tref.tolist())
    try: s['weight'] = common.deep_weight(A)
    except Exception as ex: s['weight_err'] = repr(ex)[:100]
    json.dump(s, open(os.path.join(common.DATA, ('di4_spores_b12.json' if AVP else 'di4_spores.json')), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
