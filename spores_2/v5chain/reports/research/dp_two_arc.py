"""research-10: связь двумя дугами f → P → (b + s·n_b): 7 неизвестных (t1, τ1, τ2, t2, τ1', τ2', s), 4 уравнения — минимально-нормальный
Гаусс–Ньютон с ограничениями (клип |τ| ≤ 1, t ∈ [.1, TL]). Успех ⇒ P — новая спора (нормаль = перенос n_b назад по дуге 2): центр f → центр P
дугой 1, узлы P → отрезок b дугой 2. Проверка: доля успехов на ближайших парах «прямое — обратное дерево» атласа g = 2."""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, TL
from scipy.spatial import cKDTree
from butterfly_dp_atlas import R
def two_arc(f, b, nb, x0, it=30):
    x = x0.copy()                                                                            # t1, a1, a2, t2, c1, c2, s
    def F(x):
        P = flow(f[:, None], np.array([x[1]]), np.array([x[2]]), x[0])[:, 0]; Z = flow(P[:, None], np.array([x[4]]), np.array([x[5]]), x[3])[:, 0]
        r = Z - (b + x[6] * nb); r[:2] = wrap(r[:2]); return r, P
    lo = np.array([.1, -1, -1, .1, -1, -1, -R]); hi = np.array([TL, 1, 1, TL, 1, 1, R])
    for _ in range(it):
        r, P = F(x)
        if np.abs(r).max() < 1e-9: break
        J = np.stack([(F(x + 1e-6 * e)[0] - r) / 1e-6 for e in np.eye(7)], 1); dx = -np.linalg.pinv(J) @ r; x = np.clip(x + dx, lo, hi)
    r, P = F(x); return (np.abs(r).max() < 1e-7), x, P
if __name__ == '__main__':
    import butterfly_dp_query as Q, butterfly_dp_ellipse as E
    A = Q.load(sys.argv[1]); g = E.g_from_start(A).min(1); V = A.V.min(1); fs, bs = np.flatnonzero(np.isfinite(g)), np.flatnonzero(V < 500)
    d, i = cKDTree(A.X[bs]).query(A.X[fs]); o = np.argsort(d)[:int(os.environ.get('M', 40))]; rng = np.random.default_rng(0); ok_n = 0; t0 = time.time()
    for j in o:
        f, b, nb = A.C[fs[j]], A.C[bs[i[j]]], A.n[bs[i[j]]]
        for k in range(6):
            x0 = np.r_[rng.uniform(.3, 1.2), rng.uniform(-1, 1, 2), rng.uniform(.3, 1.2), rng.uniform(-1, 1, 2), 0.]
            ok, x, P = two_arc(f, b, nb, x0)
            if ok: ok_n += 1; break
    print(json.dumps(dict(atlas=sys.argv[1][-16:], pairs=len(o), connected=ok_n, sec=round(time.time() - t0))), flush=True)
