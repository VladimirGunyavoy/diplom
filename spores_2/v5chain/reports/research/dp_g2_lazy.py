"""research-11: g = 2 — «ленивые узлы». Диагноз: рёбра из спор без V в споры с V ЕСТЬ (сотни), но приход попадает между узлами отрезка, из которых
хотя бы один BIG ⇒ интерполяция V невозможна, связь теряется. Лечение: точка прихода p = c_k + s·n_k становится новой спорой (центр p, нормаль n_k),
из её центра ищутся пары обычным образом; ребро «источник → центр новой споры» точное (a = 0). Раунды, пока появляются новые конечные V.
Запуск: G=2 WIN=1.0 python3 dp_g2_lazy.py atlas.npz"""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from butterfly_dp import BIG
from butterfly_dp_atlas import R, MN, SNAPA, snap
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; A.solve(); KS = 81; t0 = time.time(); CAP = int(os.environ.get('CAP', 4000)); OCP = float(os.environ.get('OCP', 7.636)); FROMSTART = int(os.environ.get('FROMSTART', 0))
print(json.dumps(dict(round=0, spores=int(A.K), V=round(float(A.V[KS, MN // 2]), 3), centers_fin=int((A.V[:, MN // 2] < BIG / 2).sum()))), flush=True)
for r in range(1, int(os.environ.get('ROUNDS', 8)) + 1):
    iy, k, j0, a, t = A.e; V = A.V.reshape(-1); V0, V1 = V[k * MN + j0], V[k * MN + j0 + 1]; a_ = snap(a)
    bad = ((V0 >= BIG / 2) & (a_ < 1 - SNAPA)) | ((V1 >= BIG / 2) & (a_ > SNAPA)); fin = (A.V < BIG / 2).any(1)
    src_ok = np.isfinite(E.g_from_start(A).reshape(-1))[iy] if FROMSTART else True                        # FROMSTART: только мостики — источник достижим от старта
    cand = np.flatnonzero((V[iy] >= BIG / 2) & bad & fin[k] & src_ok); s = A.sn[j0[cand]] + a[cand] * (2 * R / (MN - 1))
    _, u = np.unique(np.c_[k[cand], np.round(s / .01)], axis=0, return_index=True)                      # одна новая спора на (спора, s с шагом .01)
    key = {}; 
    if len(u) > CAP: u = np.random.default_rng(r).choice(u, CAP, replace=False)
    if not len(u): print(json.dumps(dict(round=r, new=0))); break
    if FROMSTART and A.V[KS, MN // 2] < BIG / 2: break
    kk, ss = k[cand][u], s[u]; K0 = A.K; newC = A.C[kk] + ss[:, None] * A.n[kk]; lab = {(int(x), int(round(y / .01))): K0 + i for i, (x, y) in enumerate(zip(kk, ss))}
    tgt = np.array([lab.get((int(x), int(round(y / .01))), -1) for x, y in zip(k[cand], s)]); m = tgt >= 0
    A.C = np.r_[A.C, newC]; A.n = np.r_[A.n, A.n[kk]]; A.K = len(A.C); A.V = np.r_[A.V, np.full((A.K - K0, MN), BIG)]; E.refresh(A); A.V[A.ingoal] = 0.
    out = A.pairs(A.C[K0:], np.arange(K0, A.K)); e_new = E.edges_add(A, K0 * MN + MN // 2 + MN * np.arange(A.K - K0), out)
    e_src = [iy[cand][m], tgt[m], np.full(m.sum(), MN // 2), np.zeros(m.sum()), t[cand][m]]
    e = [np.r_[x, y, z] for x, y, z in zip(A.e, e_new, e_src)]; o = np.argsort(e[0], kind='stable'); A.e = [x[o] for x in e]; A.solve()
    print(json.dumps(dict(round=r, cand=int(len(cand)), new=int(A.K - K0), new_fin=int((A.V[K0:, MN // 2] < BIG / 2).sum()), spores=int(A.K), V=round(float(A.V[KS, MN // 2]), 3),
                          centers_fin=int((A.V[:, MN // 2] < BIG / 2).sum()), sec=round(time.time() - t0))), flush=True)
np.savez(sys.argv[1].replace('.npz', '_lazy.npz'), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
if A.V[KS, MN // 2] < BIG / 2:
    A.E = Q.node_edges(A); A.nplan = 0; T, arcs, wm = Q.rollout_edges(A, KS, 0.)
    print(json.dumps(dict(V=round(float(A.V[KS, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / OCP), 4), arcs=arcs, wmax=round(float(wm), 2))), flush=True)
