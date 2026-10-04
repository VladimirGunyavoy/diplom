"""research-11: g = 2 — где на эталонном пути OCP (только для ОЦЕНКИ, не в методе) нет покрытия деревьями. Для точек пути: расстояние в вложении
до ближайшей споры прямого дерева (g < ∞) и обратного (V < BIG), их g / V, энергия. Запуск: G=2 WIN=1.0 python3 dp_g2_gap.py atlas.npz"""
import numpy as np, sys, os, json
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, G, MS, S11, S12, S22, BIG
from scipy.spatial import cKDTree
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
from butterfly_dp_grow import emb
def energy(z):
    q1, q2, w1, w2 = z.T; th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2
    return .5 * S11 * d1 ** 2 + .5 * S22 * d2 ** 2 + S12 * np.cos(th1 - th2) * d1 * d2 + G * (MS[0] * np.sin(th1) + MS[1] * np.sin(th2))
a = np.load('refdp_G2_W3_N80.npy'); T = a[0]; u = a[1:].reshape(80, 2); h = T / 80
z = np.array([-np.pi / 2, 0, 0, 0.]); P = [z]
for k in range(80):
    z = flow(z[:, None], u[k:k + 1, 0], u[k:k + 1, 1], h, n=8)[:, 0]; P.append(z)
P = np.array(P); P[:, :2] = wrap(P[:, :2]); tt = np.arange(81) * h
A = Q.load(sys.argv[1]); g = E.g_from_start(A).min(1); V = A.V.min(1); fs, bs = np.flatnonzero(np.isfinite(g)), np.flatnonzero(V < BIG / 2)
Ea = energy(A.C); print(json.dumps(dict(T=round(float(T), 3), K=int(A.K), fwd=len(fs), bwd=len(bs), both=int(np.isin(fs, bs).sum()),
      E_fwd=np.round(np.quantile(Ea[fs], [0, .5, .9, 1]), 2).tolist(), E_bwd=np.round(np.quantile(Ea[bs], [0, .1, .5, 1]), 2).tolist(),
      g_max=round(float(g[fs].max()), 2), V_max=round(float(V[bs].max()), 2))))
df, jf = cKDTree(A.X[fs]).query(emb(P)); db, jb = cKDTree(A.X[bs]).query(emb(P)); da, _ = cKDTree(A.X).query(emb(P))
print(' t     q1    q2    w1    w2    u1   u2 |   E   | d_fwd  g   | d_bwd  V   | d_any')
for k in range(0, 81, 2):
    print('%5.2f %5.2f %5.2f %5.2f %5.2f %4.1f %4.1f | %5.2f | %.2f %5.2f | %.2f %5.2f | %.2f' % (tt[k], *P[k], *(u[min(k, 79)]), energy(P[k:k + 1])[0], df[k], g[fs[jf[k]]], db[k], V[bs[jb[k]]], da[k]))
