# research-18: ускорение HexIdx._test (Ньютон обратной полилинейной карты) — ранний выход сошедшихся + F, J свёрткой по осям 2^N-тензора. Сверка с исходным.
import os, sys, time, pickle, numpy as np
sys.path.insert(0, os.path.expanduser('~/claude-work/projects/spore/spores_2/v7/src/cells7'))
import growN as G
N = G.N
def contract(Xa, sa, j=-1):
    """Xa (K, 2^N, N) → значение (K, N) полилинейной карты в sa (K, N); j ≥ 0 — производная по оси j. Вершины в порядке VOFF (ось 0 — старшая)."""
    T = Xa.reshape((len(Xa),) + (2,) * N + (N,))
    for k in range(N):                                   # сворачиваем старшую оставшуюся ось
        T = (T[:, 1] - T[:, 0]) if k == j else T[:, 0] + sa[:, k].reshape((-1,) + (1,) * (N - k)) * (T[:, 1] - T[:, 0])
    return T
def _test_fast(s, Y, pi, vv, tolc=1e-10):
    hid = vv // G.KSH; y = Y[pi] - G.SH[vv % G.KSH]; ok = ((y >= s.LO[hid]) & (y <= s.HI[hid])).all(1); pi, hid, y = pi[ok], hid[ok], y[ok]
    z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, N)))
    if not len(pi): return z
    X = s.X[hid]; sc = np.full((len(pi), N), .5); alive = np.arange(len(pi)); done = np.zeros(len(pi), bool)
    for it in range(8):
        Xa, ya, sa = X[alive], y[alive], sc[alive]
        F = contract(Xa, sa) - ya; J = np.stack([contract(Xa, sa, j) for j in range(N)], 2)
        try: d = np.linalg.solve(J, F[..., None])[..., 0]
        except np.linalg.LinAlgError: d = np.linalg.solve(J + 1e-12 * np.eye(N), F[..., None])[..., 0]
        sc[alive] = sa - d; conv = np.abs(d).max(1) < tolc
        if it in (0, 2):
            lim = 1.0 if it == 0 else .2; keep = ((sc[alive] >= -lim) & (sc[alive] <= 1 + lim)).all(1)
        else: keep = np.ones(len(alive), bool)
        done[alive[keep & conv]] = True; drop = alive[~keep]; alive = alive[keep & ~conv]
        if it in (0, 2): done[drop] = False
        if not len(alive): break
    sel = np.flatnonzero(done | np.isin(np.arange(len(pi)), alive)); sc = sc[sel]; hid, X, y, pi = hid[sel], X[sel], y[sel], pi[sel]
    F = contract(X, sc) - y; tol = 1e-7
    k = (np.linalg.norm(F, axis=1) < 1e-7) & ((sc >= -tol) & (sc <= 1 + tol)).all(1)
    return pi[k], hid[k], np.clip(sc[k], 0, 1)
if __name__ == '__main__':
    pk = 'atlas_manip60.pkl'
    if os.path.exists(pk): A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(pk, 'rb')); A.finish()
    else: A = G.Atlas(); pickle.dump(A.layers, open(pk, 'wb'))
    Y = G.step(A.P, G.US[0]); t0 = time.time(); r0 = A.stencils(Y); t1 = time.time() - t0
    orig = G.HexIdx._test; G.HexIdx._test = _test_fast; del A.qx; t0 = time.time(); r1 = A.stencils(Y); t2 = time.time() - t0; G.HexIdx._test = orig
    same = len(r0[0]) == len(r1[0]) and all(np.array_equal(a, b) for a, b in zip(r0[:2], r1[:2])) and np.allclose(r0[2], r1[2], atol=1e-6)
    print('исходный', round(t1, 1), 'с, найдено', len(r0[0]), '| быстрый', round(t2, 1), 'с, найдено', len(r1[0]), '| совпадает', same, '| ×', round(t1 / t2, 2), flush=True)
    if not same:
        a = set(zip(r0[0].tolist(), r0[1].tolist())); b = set(zip(r1[0].tolist(), r1[1].tolist())); print('только в исх.', len(a - b), 'только в быстром', len(b - a))
