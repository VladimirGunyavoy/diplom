"""Коридор top-K + кандидаты-промахи на маятнике u = .3 (2 слоя, rk4): переносится ли результат дд 3D (multiquery_corridor.md).
Запросы и мелкая V — как tests/check_nd_pend.py. Запуск из spores_2/v6: python3 ../v5chain/reports/research/corridor_pend.py [NB NF K]"""
import sys, os, json, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value, _tree, _fpath, tree_query
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from corridor_nd import refine, merge
NB, NF, K = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (400, 50, 5)
u = 0.3; tau = 0.25; Rg = 0.5; f = pendulum(1.0); rho = 0.05; kn = 8; dt = tau / 2
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
g = lambda x: np.array([Rg ** 2 - dth(x[0]) ** 2 - x[1] ** 2])
seeds = [(np.pi + Rg * 0.999 * np.cos(a), Rg * 0.999 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]   # затравки на границе окна (pend_back_seeds.py): из точки равновесия дерево вырождается
S = SysN(fl, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, seeds, per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
Vf = [F.value(x) for x in Q]


def candidates(x0, back):
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
                    miss = min(miss, max(gd(X) - Rg, 0))
                if hit: break
            out.append((t if hit else t + miss, fp + [(s2, tau) for s2 in seqs[b]]))
    return sorted(out, key=lambda c: c[0])


back = build_back(S, tau, NB, rho); rows = []
for q, (x, vf) in enumerate(zip(Q, Vf)):
    t0 = time.time(); V = replay_value(S, tau, x, NB, NF, rho, rho, back=back)[0]
    C = candidates(x, back); seen = []; best = np.inf; bs = None
    for sc, p in C:
        sq = tuple(merge(p)[0])
        if sq in seen: continue
        seen.append(sq); T, sqr, d, ok = refine(fl, x, p, g, tries=1)
        if ok and T < best: best = T; bs = (list(map(int, sqr)), list(map(float, d)))
        if len(seen) >= K: break
    # независимая проверка: мелкий rk4 (dt 0.001) по найденному коридору
    chk = None
    if bs:
        X = np.array(x, float)
        for s, d in zip(*bs): X = rk4(f, X, u * (1 if s == 0 else -1), d, dt_max=0.001)
        chk = float(gd(X))
    rows.append(dict(q=q, V_fine=vf, V_replay=V / vf, T_corr=best / vf, end_dist=chk, seq=bs[0] if bs else None, sec=time.time() - t0)); print(rows[-1], flush=True)
T = np.array([r['T_corr'] for r in rows]); V = np.array([r['V_replay'] for r in rows])
print('маятник NB %d NF %d K %d: проигрыш V/Vf mean %.3f max %.3f; коридор T/Vf mean %.3f min %.3f max %.3f; допустимо %d/%d' % (NB, NF, K, V[np.isfinite(V)].mean(), V[np.isfinite(V)].max(), T[np.isfinite(T)].mean(), T[np.isfinite(T)].min(), T[np.isfinite(T)].max(), np.isfinite(T).sum(), len(T)))
json.dump(dict(NB=NB, NF=NF, K=K, rows=rows), open(os.path.join(HERE, 'corridor_pend_%d_%d.json' % (NB, NF)), 'w'), indent=1)
