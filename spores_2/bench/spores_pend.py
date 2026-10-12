"""C2 маятник u .3: v7 grow_cells2d (атлас спор-клеток по слоям u, граф V, агент «жадный по V*/tgoal»). Запуск (aida, система python3, из bench/): VARIANT=84|56 python3 spores_pend.py
84 — по времени (SIDEOWN MADAPT .001 MROW STOL LOOK 0); 56 — по T (MADAPT .001 LOOK 20). Цель RHO .1 как в bench. Траектории записываются (u_k, dt_k) → единый проверяльщик.
Реплика rollout() с записью управлений (оригинал не меняем)."""
import os, sys, time, json
V = os.environ.get('VARIANT', '84'); base = dict(SYS='pend', TMAX='8', RMAX='.3', OVL='.05', RHO='.1', DELTA='1e9', GOALB='1', GROW='2', OVH='.5', FRAC='.5', M='5', DTN='.06', NORM='gram', SEL='stop', SELFOV='1', BFINE='8', TQDM_MI='1000')
base.update(dict(SIDEOWN='1', MADAPT='.001', MROW='1', STOL='1e-6', LOOK='0') if V == '84' else dict(MADAPT='.001', LOOK='20'))
for k, v in base.items(): os.environ.setdefault(k, v)
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, common
import grow_cells2d as G

def rollout_rec(A, Q, tmax=30.):
    """как Atlas.rollout, но возвращает ещё U[k], H[k], act[k] по шагам (управление и длительность шага)"""
    Y = np.array(Q, float); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); UA = np.array(G.US); rec = []
    for _ in range(int(tmax / G.DTN)):
        if done.all(): break
        if G.QFAST: tgs = A.tgoal3(Y); Ys3 = A.step3(Y); J = np.minimum(tgs, G.DTN + (A.look(Ys3, G.LOOK) if G.LOOK else A.vstar(Ys3)).reshape(len(G.US), n).T)
        else: tgs = np.stack([A.tgoal(Y, u) for u in G.US], 1); J = np.minimum(tgs, G.DTN + np.stack([(A.look(G.step(Y, u), G.LOOK) if G.LOOK else A.vstar(G.step(Y, u))) for u in G.US], 1))
        k = J.argmin(1); u = UA[k]; tg = tgs[np.arange(n), k]; stuck = J.min(1) >= G.BIG / 2; act = ~done & ~stuck; Yn = Y * 0.
        for ui in G.US:
            m = act & (u == ui)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], G.DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, G.DTN / 8, 1); y8 = np.where((hh >= G.DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        h = np.where(np.isfinite(tg), tg, G.DTN); rec.append((u.copy(), h.copy(), act.copy())); T += np.where(act, h, 0.); T[~done & stuck] = np.inf; Y = np.where(act[:, None], Yn, Y); done |= G.ingoal(Y) | stuck
    T[~G.ingoal(Y)] = np.inf; return T, rec

if __name__ == '__main__':
    e = common.make('pend'); t0 = time.perf_counter(); A = G.Atlas(); t_grow = time.perf_counter() - t0; t0 = time.perf_counter(); A.solve(); t_solve = time.perf_counter() - t0
    Q = e.X0; res = []; Ts = []; tq = []
    for i in range(len(Q)):
        t1 = time.perf_counter(); T, rec = rollout_rec(A, Q[i:i + 1]); dtq = time.perf_counter() - t1; tq.append(dtq)
        us = [[r[0][0]] for r in rec if r[2][0]]; dts = [float(r[1][0]) for r in rec if r[2][0]]
        Tv, why = common.verify(e, Q[i], us, dts); Ts.append(Tv); res.append(dict(i=i, T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_query=dtq, nsteps=len(us)))
    t1 = time.perf_counter(); Tb, _ = rollout_rec(A, Q); t_batch = time.perf_counter() - t1
    s = common.summarize(e, Ts); s.update(env='pend', method='spores v7 grow_cells2d ' + ('84 (по времени: LOOK 0)' if V == '84' else '56 (по T: LOOK 20)'), t_grow=t_grow, t_solve=t_solve, t_build=t_grow + t_solve, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)), t_batch20=t_batch,
                                      cells=[len(l) for l in A.layers], nodes=int(A.N), hw='CPU (1 proc; numpy)', per_start=res, Tref=e.Tref.tolist())
    json.dump(s, open(os.path.join(common.DATA, 'pend_spores%s.json' % ('' if V == '84' else '_T')), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
