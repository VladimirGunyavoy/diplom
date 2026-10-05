"""research-13: почему прямое (от старта) и обратное (от цели) деревья не связываются. Вход: атлас.npz, эталонный путь ocp_arcs (.npy: T, U (K×2), H (K)), X0.
(1) вдоль эталонного пути: расстояние (вложение) до ближайшей прямой / обратной споры, энергия; (2) кандидаты-пары «узел прямой споры → обратная спора»
в окне WIN и причины отказа solve_arcs. Запуск: G=2 WIN=1.0 WPAIR=1 X0=q1,q2,w1,w2 python3 dp_gap_diag.py atlas.npz ref.npy"""
import numpy as np, sys, os, json
sys.path.insert(0, '.')
import butterfly_dp as D
from butterfly_dp import flow, wrap, ingoal, BIG, WIN, WMAX, TL, G, solve_arcs, S11, S12, S22, MS
from butterfly_dp_atlas import R, MN
from scipy.spatial import cKDTree
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
from butterfly_dp_grow import emb
def energy(z):
    d1, d2 = z[2], z[2] + z[3]; c = np.cos(z[1])
    return .5 * (S11 * d1 ** 2 + S22 * d2 ** 2 + 2 * S12 * c * d1 * d2) + G * (MS[0] * np.sin(z[0]) + MS[1] * np.sin(z[0] + z[1]))
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; A.solve(); KS = E.KS
g = E.g_from_start(A).min(1); V = A.V.min(1); F = np.flatnonzero(np.isfinite(g)); B = np.flatnonzero(V < BIG / 2); both = np.intersect1d(F, B)
print(json.dumps(dict(spores=int(A.K), fwd=len(F), bwd=len(B), both=len(both), neither=int(A.K - len(np.union1d(F, B))), V_start=round(float(A.V[KS, MN // 2]), 3))), flush=True)
EF, EB = energy(A.C[F].T), energy(A.C[B].T); pc = lambda x: [round(float(v), 2) for v in np.percentile(x, [0, 10, 50, 90, 100])]
print(json.dumps(dict(E_fwd=pc(EF), E_bwd=pc(EB), E_start=round(float(energy(A.C[KS])), 2), E_goal=round(float(energy(np.array([np.pi / 2, 0, 0, 0.]))), 2),
                      absw_fwd=pc(np.abs(A.C[F, 2:]).max(1)), absw_bwd=pc(np.abs(A.C[B, 2:]).max(1)))), flush=True)
tF, tB = cKDTree(A.X[F]), cKDTree(A.X[B]); dFB = tB.query(A.X[F])[0]
print(json.dumps(dict(dist_fwd_to_nearest_bwd=pc(dFB), fwd_within_0p3_of_bwd=int((dFB < .3).sum()))), flush=True)
# (1) эталонный путь
if len(sys.argv) > 2:
    d = np.load(sys.argv[2]); K = (len(d) - 1) // 3; U = d[1:1 + 2 * K].reshape(K, 2); H = d[1 + 2 * K:]; z = np.array([float(v) for v in os.environ['X0'].split(',')]); t = 0.; rows = []; nxt = 0.
    for u, h in zip(U, H):
        n = max(1, int(np.ceil(h / .002))); hh = h / n
        for _ in range(n):
            if t >= nxt - 1e-9:
                zz = z.copy(); zz[:2] = wrap(zz[:2]); x = emb(zz[None]); rows.append((t, tF.query(x)[0][0], tB.query(x)[0][0], energy(z), np.abs(z[2:]).max(), *zz)); nxt += .25
            z = flow(z[:, None], np.array([u[0]]), np.array([u[1]]), hh, n=1)[:, 0]; t += hh
    print('t      dF     dB     E      |w|   q1     q2     w1     w2   (путь OCP, T %.3f, конец в цели: %s)' % (t, bool(ingoal(z[:, None])[0])))
    for r in rows: print('%5.2f  %5.2f  %5.2f  %6.2f  %4.2f  %5.2f  %5.2f  %5.2f  %5.2f' % r)
# (2) пары: узлы прямых спор → обратные споры в окне
rng = np.random.default_rng(0); Fs = rng.choice(F, min(len(F), 1500), replace=False); Y = A.P[Fs].reshape(-1, 4); own = np.repeat(Fs, MN); isB = np.zeros(A.K, bool); isB[B] = True
nb = A.tree.query_ball_point(emb(Y), 2 * WIN); iy = np.repeat(np.arange(len(Y)), [len(b) for b in nb]); k = np.concatenate([np.asarray(b, int) for b in nb])
m = isB[k] & (k != own[iy]); iy, k = iy[m], k[m]; dd = np.c_[wrap(A.C[k, :2] - Y[iy, :2]), A.C[k, 2:] - Y[iy, 2:]]; m = (np.abs(dd) <= WIN).all(1); iy, k = iy[m], k[m]
st = dict(cand=int(len(iy)))
if len(iy):
    res = [solve_arcs(Y[iy[a:a + 200000]].T, A.C[k[a:a + 200000]].T, A.n[k[a:a + 200000]].T) for a in range(0, len(iy), 200000)]; t, u1, u2, sv, ok = [np.concatenate(x) for x in zip(*res)]
    Z = np.concatenate([flow(Y[iy[a:a + 200000]].T, u1[a:a + 200000], u2[a:a + 200000], t[a:a + 200000]) for a in range(0, len(iy), 200000)], 1)
    Ct = A.C[k].T.copy(); Ct[:2] = Y[iy, :2].T + wrap(A.C[k, :2] - Y[iy, :2]).T; conv = np.abs(np.nan_to_num(Z - (Ct + sv * A.n[k].T), nan=9.)).max(0) < 1e-7
    tok = (t > 1e-3) & (t <= TL); uok = (np.abs(u1) <= 1 + 1e-9) & (np.abs(u2) <= 1 + 1e-9); sok = np.abs(sv) <= R
    st.update(converged=int(conv.sum()), conv_t_ok=int((conv & tok).sum()), conv_t_u_ok=int((conv & tok & uok).sum()), conv_t_u_s_ok=int((conv & tok & uok & sok).sum()), ok_and_s=int((ok & sok).sum()),
              conv_t_s_ok_u_bad=int((conv & tok & sok & ~uok).sum()), umax_of_those=pc(np.maximum(np.abs(u1), np.abs(u2))[conv & tok & sok & ~uok]) if (conv & tok & sok & ~uok).any() else None)
    gd = np.flatnonzero(ok & sok)
    if len(gd):                                                                           # годные пары есть — почему V не проходит: |w| вдоль дуги и BIG у узлов прихода
        wok = np.ones(len(gd), bool)
        for fr in (.2, .4, .6, .8): zz = flow(Y[iy[gd]].T, u1[gd], u2[gd], t[gd] * fr); wok &= (np.abs(np.nan_to_num(zz[2:], nan=99.)) <= WMAX).all(0)
        f_ = (sv[gd] + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); V0, V1 = A.V[k[gd], j0], A.V[k[gd], j0 + 1]
        st.update(good_w_ok=int(wok.sum()), good_w_ok_both_nodes_fin=int((wok & (V0 < BIG / 2) & (V1 < BIG / 2)).sum()), good_w_ok_any_node_fin=int((wok & ((V0 < BIG / 2) | (V1 < BIG / 2))).sum()))
print(json.dumps(st), flush=True)
