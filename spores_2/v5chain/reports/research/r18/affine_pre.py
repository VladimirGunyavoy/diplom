# research-18: аффинный предфильтр стенсилов — центр A0 и J⁻¹ в центре на гиперячейку (заранее), отбор пар по sc0 = .5 + J⁻¹(y − A0) ∈ [−MG, 1+MG], Ньютон от sc0.
import os, sys, time, pickle, numpy as np
import fast_test as FT
G = FT.G; N = G.N; MG = float(os.environ.get('MG', .25))
def prep(s):
    if getattr(s, '_pn', -1) == s.n: return
    X = s.X[:s.n]; h = np.full((s.n, N), .5); s.A0 = FT.contract(X, h); J = np.stack([FT.contract(X, h, j) for j in range(N)], 2); s.JI = np.linalg.pinv(J); s._pn = s.n
def _test_aff(s, Y, pi, vv, tolc=1e-10):
    prep(s); hid = vv // G.KSH; y = Y[pi] - G.SH[vv % G.KSH]; ok = ((y >= s.LO[hid]) & (y <= s.HI[hid])).all(1); pi, hid, y = pi[ok], hid[ok], y[ok]
    sc0 = .5 + np.einsum('kij,kj->ki', s.JI[hid], y - s.A0[hid]); ok = ((sc0 >= -MG) & (sc0 <= 1 + MG)).all(1); pi, hid, y, sc = pi[ok], hid[ok], y[ok], sc0[ok]
    z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, N)))
    if not len(pi): return z
    X = s.X[hid]; alive = np.arange(len(pi))
    for it in range(8):
        Xa, ya, sa = X[alive], y[alive], sc[alive]; F = FT.contract(Xa, sa) - ya; J = np.stack([FT.contract(Xa, sa, j) for j in range(N)], 2)
        try: d = np.linalg.solve(J, F[..., None])[..., 0]
        except np.linalg.LinAlgError: d = np.linalg.solve(J + 1e-12 * np.eye(N), F[..., None])[..., 0]
        sc[alive] = sa - d; keep = ((sc[alive] >= -.2) & (sc[alive] <= 1.2)).all(1) if it >= 1 else np.ones(len(alive), bool)
        bad = alive[~keep]; sc[bad] = 9.; alive = alive[keep & (np.abs(d).max(1) >= tolc)]
        if not len(alive): break
    F = FT.contract(X, sc.clip(-1, 2)) - y; tol = 1e-7
    k = (np.linalg.norm(F, axis=1) < 1e-7) & ((sc >= -tol) & (sc <= 1 + tol)).all(1)
    return pi[k], hid[k], np.clip(sc[k], 0, 1)
if __name__ == '__main__':
    A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open('atlas_manip60.pkl', 'rb')); A.finish(); Y = G.step(A.P, G.US[0])
    t0 = time.time(); r0 = A.stencils(Y); t1 = time.time() - t0
    G.HexIdx._test = _test_aff; del A.qx; t0 = time.time(); r1 = A.stencils(Y); t2 = time.time() - t0
    a = set(zip(r0[0].tolist(), r0[1][:, 0].tolist())); b = set(zip(r1[0].tolist(), r1[1][:, 0].tolist()))
    print('MG', MG, '| исходный', round(t1, 1), 'с', len(a), '| аффинный', round(t2, 1), 'с', len(b), '| потеряно', len(a - b), 'лишних', len(b - a), '| ×', round(t1 / t2, 2), flush=True)
