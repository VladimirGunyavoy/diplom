"""research-10: сколько пар теряет Ньютон butterfly_dp.solve_arcs из-за старта (формула ДИ) — против нескольких стартов τ и t.
Кандидаты — (узел, спора) в окне WIN выращенного атласа; годная пара — сходимость, t ≤ TL, |τ| ≤ 1, |s| ≤ R."""
import numpy as np, os, sys, json
sys.path.insert(0, '.'); os.environ.setdefault('WIN', '1.0')
import butterfly_dp as D
from butterfly_dp import wrap, flow, f, TL, WIN
import butterfly_dp_query as Q
from butterfly_dp_atlas import R, MN
def newton(Y, Ct, Nn, t, u1, u2, s, it=10):
    for _ in range(it):
        Z = flow(Y, u1, u2, t); Rz = Z - (Ct + s * Nn); e = 1e-5; Z1 = flow(Y, u1 + e, u2, t); Z2 = flow(Y, u1, u2 + e, t)
        J = np.moveaxis(np.stack([f(Z, u1, u2), (Z1 - Z) / e, (Z2 - Z) / e, -Nn], -1), 1, 0); rhs = np.moveaxis(Rz, 0, 1)[..., None]
        bad = ~np.isfinite(J).all((1, 2)) | ~np.isfinite(rhs).all((1, 2)) | (np.abs(np.linalg.det(np.nan_to_num(J))) < 1e-12); J[bad] = np.eye(4); rhs[bad] = 0
        d = np.linalg.solve(J, rhs)[..., 0]; t = np.clip(t - d[:, 0], 1e-4, 2 * TL); u1 = np.clip(u1 - d[:, 1], -3, 3); u2 = np.clip(u2 - d[:, 2], -3, 3); s = s - d[:, 3]
    res = np.abs(flow(Y, u1, u2, t) - (Ct + s * Nn)).max(0)
    return (res < 1e-7) & (t > 1e-3) & (t <= TL) & (np.abs(u1) <= 1 + 1e-9) & (np.abs(u2) <= 1 + 1e-9) & (np.abs(s) <= R), t
A = Q.load(sys.argv[1]); rng = np.random.default_rng(0)
P = (A.C[:, None, :] + A.sn[None, :, None] * A.n[:, None, :]).reshape(-1, 4); own = np.repeat(np.arange(A.K), MN)
iy = rng.choice(len(P), 1500, replace=False); nb = A.tree.query_ball_point(Q.emb(P[iy]), 2 * WIN)
I = np.repeat(iy, [len(b) for b in nb]); K = np.concatenate([np.asarray(b, int) for b in nb])
d = np.c_[wrap(A.C[K, :2] - P[I, :2]), A.C[K, 2:] - P[I, 2:]]; g = (np.abs(d) <= WIN).all(1) & (K != own[I]); I, K = I[g], K[g]
Y, C, Nn = P[I].T, A.C[K].T, A.n[K].T; Ct = C.copy(); Ct[:2] = Y[:2] + wrap(C[:2] - Y[:2])
_, _, _, s0, ok0 = D.solve_arcs(Y, C, Nn); ok0 &= np.abs(s0) <= R; okany = ok0.copy(); out = dict(cands=len(I), cand_per_node=round(len(I) / 1500, 1), formula=int(ok0.sum()))
for tt in (.3, .7, 1.2):
    for a, b in ((0, 0), (.8, .8), (.8, -.8), (-.8, .8), (-.8, -.8)):
        n = len(I); ok, _ = newton(Y, Ct, Nn, np.full(n, tt), np.full(n, a, float), np.full(n, b, float), np.zeros(n)); okany |= ok
    out['after_t%.1f' % tt] = int(okany.sum())
out['gain'] = round(okany.sum() / max(ok0.sum(), 1), 2); print(json.dumps(out), flush=True)
