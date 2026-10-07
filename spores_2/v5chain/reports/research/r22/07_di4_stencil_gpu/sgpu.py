"""research-22, эксп. 07: HexIdx._test (обратная полилинейная карта Ньютоном — 81% времени стенсилов, эксп. 06) на GPU (torch, float64).
Логика 1:1 с growN._test (FASTT=1); подмена метода. Проверка: множество пар (точка, гиперячейка) и веса — как у CPU. di4 L400, один u."""
import sys, os, time, pickle, numpy as np, torch
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
N = G.N; dev = 'cuda'; CH = int(os.environ.get('GCH', 1 << 21)); T = lambda a: torch.as_tensor(a, device=dev)
def contract(Xa, sa, j=-1):
    t = Xa.reshape((len(Xa),) + (2,) * N + (N,))
    for k in range(N): t = (t[:, 1] - t[:, 0]) if k == j else t[:, 0] + sa[:, k].reshape((-1,) + (1,) * (N - k)) * (t[:, 1] - t[:, 0])
    return t
def test_gpu(s, Y, pi, vv, tolc=1e-10):
    s._prep()
    if getattr(s, '_gn', -1) != s.n: s._g = [T(a[:s.n] if len(a) >= s.n else a) for a in (s.X, s.LO, s.HI, s.A0, s.JI)]; s._gn = s.n; s._gsh = T(np.asarray(G.SH, float))
    if getattr(s, '_gy', None) is not Y: s._gy = Y; s._gY = T(Y)
    X_, LO, HI, A0, JI = s._g; MG = float(G.E('MG', .4)); out = []
    for a in range(0, len(pi), CH):
        p = T(pi[a:a + CH]); v = T(vv[a:a + CH]); hid = v // G.KSH; y = s._gY[p] - s._gsh[v % G.KSH]; ok = ((y >= LO[hid]) & (y <= HI[hid])).all(1); p, hid, y = p[ok], hid[ok], y[ok]
        sc = .5 + torch.einsum('kij,kj->ki', JI[hid], y - A0[hid]); ok = ((sc >= -MG) & (sc <= 1 + MG)).all(1); p, hid, y, sc = p[ok], hid[ok], y[ok], sc[ok]
        if not len(p): continue
        X = X_[hid]; alive = torch.arange(len(p), device=dev)
        for it in range(8):
            Xa, ya, sa = X[alive], y[alive], sc[alive]; F = contract(Xa, sa) - ya; J = torch.stack([contract(Xa, sa, j) for j in range(N)], 2)
            d = torch.linalg.solve_ex(J, F[..., None])[0][..., 0]; d = torch.nan_to_num(d, nan=1e9, posinf=1e9, neginf=-1e9)
            sn = sa - d; sc[alive] = sn; keep = ((sn >= -.2) & (sn <= 1.2)).all(1) if it >= 1 else torch.ones(len(alive), dtype=torch.bool, device=dev)
            sc[alive[~keep]] = 9.; alive = alive[keep & (d.abs().max(1).values >= tolc)]
            if not len(alive): break
        F = contract(X, sc.clip(-1, 2)) - y; tol = 1e-7; k = (torch.linalg.norm(F, dim=1) < 1e-7) & ((sc >= -tol) & (sc <= 1 + tol)).all(1)
        out.append((p[k].cpu().numpy(), hid[k].cpu().numpy(), sc[k].clip(0, 1).cpu().numpy()))
    if not out: return np.zeros(0, int), np.zeros(0, int), np.zeros((0, N))
    return tuple(np.concatenate(o) for o in zip(*out))
if __name__ == '__main__':
    d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); Y = G.step(A.P, G.US[0]); cpu_test = G.HexIdx._test
    torch.zeros(1, device=dev); res = {}
    for name, fn in (('GPU', test_gpu), ('GPU повтор', test_gpu), ('CPU', cpu_test)):
        G.HexIdx._test = fn; torch.cuda.synchronize(); t0 = time.time(); a, b, c = A.stencils(Y); torch.cuda.synchronize(); dt = time.time() - t0
        o = np.lexsort((b[:, 0], a)); res[name] = (a[o], b[o], c[o]); print('%-11s stencils одного u: %.1f с, пар %d' % (name, dt, len(a)), flush=True)
    (a1, b1, c1), (a2, b2, c2) = res['GPU'], res['CPU']
    same = len(a1) == len(a2) and (a1 == a2).all() and (b1 == b2).all(); print('множество пар совпало:', bool(same), '| max|ΔW| %.3g' % (np.abs(c1 - c2).max() if same else np.nan), '| память GPU пик %.1f ГБ' % (torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    if not same:
        k1 = set(zip(a1.tolist(), b1[:, 0].tolist())); k2 = set(zip(a2.tolist(), b2[:, 0].tolist())); print('только GPU', len(k1 - k2), 'только CPU', len(k2 - k1), 'общих', len(k1 & k2))
