"""Г4.2 рука (а), ОПИСАТЕЛЬНО: карта «узел→узел» при ε=h (ε выбран ПОСЛЕ просмотра Г4.1: при h/3 SCC 56%) + Дейкстра от цели по обратным рёбрам + исполнение реальной динамикой с перепривязкой к ближайшему узлу на каждом ребре.
python g42_map.py 100000"""
import os, sys, time, json, hashlib, runpy
N_ = sys.argv[1]; sys.argv = ['g4a.py', 'di4', N_, '1.0']
g = runpy.run_path('g4a.py', run_name='lib'); import numpy as np
P, tree, EM, h, edges, reduce_eps, N, US, adv, wrap, embed, inside, diff, lim = [g[k] for k in ('P','tree','EM','h','edges','reduce_eps','N','US','adv','wrap','embed','inside','diff','lim')]
from scipy.sparse import csr_matrix; from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import g42_common
BENCH = os.path.expanduser('~/spore_v5/a31/spores_2/bench'); sys.path.insert(0, BENCH); import common
env = common.make('di4'); R = .35
print('HEADER: рука (а) карта узел→узел ДИ4D N=%d ε=h=%.4f (выбран после просмотра Г4.1), TMAX=1, |US|=9; цели/старты g42_common; проверяльщик common.verify; код md5=%s' % (N, h, hashlib.md5(open('g4a.py', 'rb').read()).hexdigest()[:10]), flush=True)
t0 = time.time(); ei, ey, et, ed, ui = edges(P, tree, np.arange(N), EM); ei, ey, et, uk = reduce_eps(ei, ey, et, ed, h, ui); del ed, ui; t_map = time.time() - t0
print('карта: рёбер %d, построение %.1f с' % (len(ei), t_map), flush=True)
B = csr_matrix((et, (ey, ei)), shape=(N, N)); out_ptr = np.searchsorted(ei, np.arange(N + 1)) if (np.diff(ei) >= 0).all() else None
o_ = np.lexsort((ey, ei)); ei, ey, et, uk = ei[o_], ey[o_], et[o_], uk[o_]; ptr = np.searchsorted(ei, np.arange(N + 1))
rows = []; allR = []; tg = []; tqs = []
for gi, (C, Q, ref, keep) in enumerate(g42_common.make()):
    t1 = time.time(); ing = (np.abs(P - C) < R - .02).all(1); gn = np.nonzero(ing)[0]; W = dijkstra(B, directed=True, indices=gn, min_only=True); tg.append(time.time() - t1); fin = np.isfinite(W); res = []
    ing_f = lambda y, C=C: (np.abs(y - C) < R).all(-1)
    for i in range(len(Q)):
        if not keep[i]: continue
        t2 = time.time(); x = Q[i].copy(); us = []; dts = []; done = False
        for step in range(80):
            if ing_f(x): done = True; break
            _, n0 = tree.query(embed(x[None]), k=1); n0 = int(n0[0]); a, b = ptr[n0], ptr[n0 + 1]
            if a == b: break
            c = et[a:b] + np.where(fin[ey[a:b]], W[ey[a:b]], 1e9); j = int(np.argmin(c))
            if c[j] >= 1e8: break
            u = US[uk[a + j]]; dt = float(et[a + j]); us.append(u); dts.append(dt); x = adv(x[None], u, np.array([dt]))[0]
        tqs.append(time.time() - t2)
        if gi == 0 and i < 3: print('  DBG цель0 старт %d: W конечных %.3f, W[n0 старта] %.3f, шагов %d, done %s, x_конец %s, Tref %.3f, сумма dt %.3f' % (i, fin.mean(), W[int(tree.query(embed(Q[i][None]), k=1)[1][0])], len(us), done, np.round(x, 2).tolist(), ref[i], sum(dts)), flush=True)
        if not us: res.append((None, 'no_path', ref[i])); continue
        Tv, why = common.verify(type('E', (), dict(in_goal=lambda s_, y, C=C: (np.abs(y - C) < R).all(-1), clip_u=env.clip_u, step=env.step, out_of_box=env.out_of_box))(), Q[i], us, dts); res.append((Tv, why, ref[i]))
    ok = [t / r for t, w, r in res if t is not None and t <= 3 * r]; allR.append(res); rows.append(dict(goal=C[:2].tolist(), n=len(res), reach=len(ok), t_goal=tg[-1], med=float(np.median(ok)) if ok else None, why={w: sum(1 for x_ in res if x_[1] == w) for w in set(x_[1] for x_ in res)}))
    print('цель %d: Дейкстра %.2f с, %s' % (gi, tg[-1], rows[-1]), flush=True)
R_ = np.array([t / r for res in allR for t, w, r in res if t is not None and t <= 3 * r]); nk = sum(r['n'] for r in rows); tq = np.array(tqs)
s = dict(n=nk, reach=int(len(R_)), med=float(np.median(R_)) if len(R_) else None, p90=float(np.percentile(R_, 90)) if len(R_) else None, max=float(R_.max()) if len(R_) else None, t_map=t_map, t_goal_med=float(np.median(tg)), t_query_med=float(np.median(tq)), edges=int(len(ei)), why_total={w: sum(r['why'].get(w, 0) for r in rows) for w in ('ok', 'no_goal', 'no_path', 'box')}, rows=rows)
print('ИТОГ (а, ε=h, описательно): ' + json.dumps({k_: v for k_, v in s.items() if k_ != 'rows'}), flush=True)
o = os.path.expanduser('~/spore_v5/c13/out_g42'); os.makedirs(o, exist_ok=True); json.dump(s, open(o + '/arm_a.json', 'w'), default=float)
