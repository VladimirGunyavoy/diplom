"""Коридор top-K топологий: из всех стыков проигрыша (каждая прямая спора × ближайшие обратные) берём K лучших по V РАЗНЫХ последовательностей
слоёв (после слияния), каждую уточняем SLSQP, итог — минимум. Гипотеза: выбросы corridor_dd (T/ref 1.28) — неверная топология лучшего стыка.
Запуск из spores_2/v6: python3 ../v5chain/reports/research/corridor_topk_dd.py [NB NF K]"""
import sys, os, json, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, _tree, _fpath, tree_query
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from corridor_nd import refine, merge
NB, NF, K = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (1200, 200, 5)
MISS = len(sys.argv) > 4 and sys.argv[4] == 'miss'   # кандидаты и с промахом: score = t + мин. расстояние до окна (SLSQP сам обеспечит вход)
tau = 0.25; R = .25; Rth = .26; rho = 0.08; kn = 8; dt = tau / 2
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)


def candidates(x0, back):
    """Все стыки: (время входа в окно, путь). Упрощённый проигрыш replay_value (без препятствий), та же сетка dt."""
    Pb, Gb, parb, layb, seqs, tree = back; Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True); out = []
    for i in range(len(Pf)):
        fp = _fpath(Pf, parf, layf, i, tau)
        if S.in_goal(Pf[i]): out.append((Gf[i], fp)); continue
        _, j = tree_query(S, tree, Pb, Pf[i], kn)
        for b in j:
            X = np.array(Pf[i]); t = Gf[i]; hit = False; miss = np.inf
            for s in seqs[b]:
                for _ in range(2):
                    X = S.wrap(S.flow(X, s, dt)); t += dt
                    if S.in_goal(X): hit = True; break
                    miss = min(miss, max(np.hypot(X[0], X[1]) - R, 0) + max(abs(dth(X[2])) - Rth, 0))
                if hit: break
            if hit: out.append((t, fp + [(s2, tau) for s2 in seqs[b]]))
            elif MISS: out.append((t + miss, fp + [(s2, tau) for s2 in seqs[b]]))
    return sorted(out, key=lambda c: c[0])


E = json.load(open(os.path.join(HERE, 'multiquery_dd_ref.json'))); M = len(E)
rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(M)]
back = build_back(S, tau, NB, rho); r1 = []; rK = []; sol = []
for q, (x, e) in enumerate(zip(Q, E)):
    C = candidates(x, back); seen = []; best1 = None; bestK = np.inf; bsol = None
    for V, p in C:
        sq = tuple(merge(p)[0])
        if sq in seen: continue
        seen.append(sq); T, sqr, dts, okr = refine(fl, x, p, g)
        if best1 is None and okr: best1 = T                      # top-1 = первая ДОПУСТИМАЯ топология
        if okr and T < bestK: bestK = T; bsol = dict(q=q, seq=list(map(int, sqr)), dts=list(map(float, dts)))
        if len(seen) >= K: break
    if best1 is None: print(q, 'нет допустимой топологии', len(seen)); continue
    r1.append(best1 / e); rK.append(bestK / e); sol.append(bsol); print(q, 'топологий', len(seen), 'T1/ref %.3f TK/ref %.3f' % (r1[-1], rK[-1]), seen[:3], flush=True)
r1, rK = np.array(r1), np.array(rK)
print(('miss ' if MISS else '') + 'NB %d NF %d K %d: допустимо %d/%d; top-1 mean %.3f max %.3f; top-%d mean %.3f max %.3f; доля ≤1.001: %d/%d' % (NB, NF, K, len(rK), M, r1.mean(), r1.max(), K, rK.mean(), rK.max(), (rK <= 1.001).sum(), len(rK)))
json.dump(dict(NB=NB, NF=NF, K=K, top1=r1.tolist(), topK=rK.tolist(), sol=sol), open(os.path.join(HERE, 'corridor_topk_dd%s.json' % ('_miss' if MISS else '')), 'w'), indent=1)
