"""research-22, эксп. 12: di4 с полными слоями (L1600 от b5, п.38) — весь конвейер после build на GPU: стенсилы (sgpu.test_gpu) → рёбра SBCAUS при SBLAY 0/1/2 →
Якоби без защёлки (PESS 1) → V к точной LB (коробка ±RHO) → агент на 60 стартах (эталон ±.35). Вопрос пользователя: при полном слое правило «переключение в слой u2» перестаёт терять?"""
import sys, os, time, pickle, numpy as np, torch
sys.path.insert(0, '.')
import growN as G, sgpu
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from tqdm import tqdm
dev = 'cuda'; DT = torch.float64; BIG = G.BIG; DTN = G.DTN; US = G.US; N4 = G.N; R = float(os.environ['RHO']); PESS = 1.; WTHR = .5; G.HexIdx._test = sgpu.test_gpu
def tbox(x0, v0, K=81):
    s_ = np.linspace(-R, R, K); tx = np.r_[s_, s_, np.full(K, -R), np.full(K, R)]; tv = np.r_[np.full(K, -R), np.full(K, R), s_, s_]; out = np.full(len(x0), np.inf)
    for a in tqdm(range(0, len(x0), 20000), desc='tbox', mininterval=10):
        x = x0[a:a + 20000, None]; v = v0[a:a + 20000, None]; best = np.full(x.shape[0], np.inf)
        for s in (1., -1.):
            q = s * (tx - x) + (v * v + tv * tv) / 2; vm = s * np.sqrt(np.maximum(q, 0)); t1 = s * (vm - v); t2 = s * (vm - tv); best = np.minimum(best, np.where((q >= 0) & (t1 >= -1e-12) & (t2 >= -1e-12), t1 + t2, np.inf).min(1))
        out[a:a + 20000] = best
    out[(np.abs(x0) <= R) & (np.abs(v0) <= R)] = 0.; return out
def gsolve(I, IDX, W, Vg, goal, N):
    gI = torch.as_tensor(I.astype(np.int64), device=dev); gX = torch.as_tensor(IDX, device=dev); gW = torch.as_tensor(W, device=dev); gg = torch.as_tensor(goal, device=dev); gVg = torch.as_tensor(Vg, device=dev).to(DT)
    V = torch.full((N,), BIG, dtype=DT, device=dev); V[gg] = 0.; V = torch.minimum(V, gVg); n = 0; ch = 1 << 21; t0 = time.time()
    while n < 3000:
        t = torch.full((N,), BIG, dtype=DT, device=dev)
        for a in range(0, len(gI), ch):
            vi = V[gX[a:a + ch].long()]; ok = vi < BIG / 2; w = gW[a:a + ch].to(DT) * ok; sm = w.sum(1); mx = torch.where(ok, vi, torch.full_like(vi, -float('inf'))).max(1).values
            v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) + torch.nan_to_num((1 - sm) * (mx + PESS), nan=0., posinf=0., neginf=0.); v[sm < WTHR] = BIG; t.scatter_reduce_(0, gI[a:a + ch], DTN + v, 'amin')
        t[gg] = 0.; t = torch.minimum(t, gVg); d = float((t - V).abs().max()); V = t; n += 1
        if d < 1e-9: break
    torch.cuda.synchronize(); out = V.cpu().numpy(); dt = time.time() - t0; del gI, gX, gW, V, t; torch.cuda.empty_cache(); return out, n, dt
def rollout(s, Q, tmax=40.):
    Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); sw = np.zeros(n, int); pk = np.full(n, -1); ar = np.arange(n)
    for _ in range(int(tmax / DTN)):
        if done.all(): break
        tgs = np.stack([s.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([s.vstar(G.step(Y, u)) for u in US], 1)); k = J.argmin(1); tg = tgs[ar, k]; stuck = J[ar, k] >= BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
        for ki, ui in enumerate(US):
            m = act & (k == ki)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        sw += act & (pk >= 0) & (k != pk); pk = np.where(act, k, pk); Y = G.wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= G.ingoal(Y) | stuck
    T[~G.ingoal(Y)] = np.inf; return T, sw
t0 = time.time(); d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); N = A.N; P = A.P
print('слои %s клеток, узлов %d, finish %.1f с' % ([len(l) for l in A.layers], N, time.time() - t0), flush=True)
cl = np.concatenate([[k] * len(l) for k, l in enumerate(A.layers)]); O = np.array([c.o for c in A.cells]); nl = cl[np.searchsorted(O, np.arange(N), 'right') - 1]; Vg = np.full(N, np.inf); ST = []; t0 = time.time()
for ui, u in enumerate(tqdm(US, desc='стенсилы', mininterval=1)):
    Vg = np.minimum(Vg, A.tgoal(P, u)); a, b, c = A.stencils(G.step(P, u)); nc = np.searchsorted(O, a, 'right') - 1; vc = np.searchsorted(O, b[:, 0], 'right') - 1; own = cl[nc] == ui; ml = cl[vc] == ui; hs = np.zeros(N, bool); hs[a[ml & ~own]] = True
    print('u %d: пар %d (%.2f на узел) | узлов чужих слоёв со стенсилом в слой u: %.3f | с любым стенсилом: %.3f' % (ui, len(a), len(a) / N, hs[nl != ui].mean(), np.isin(np.flatnonzero(nl != ui), a).mean()), flush=True)
    ST.append((a.astype(np.int32), b.astype(np.int32), c.astype(np.float32), own, ml, vc != nc, hs[a]))
ts = time.time() - t0; print('стенсилы 4 u на GPU: %.1f с' % ts, flush=True)
mm = G.M ** (N4 - 1); ea = np.concatenate([c.o + np.arange((c.G.shape[0] - 1) * mm) for c in A.cells]).astype(np.int32); ex = np.zeros((len(ea), 2 ** N4), np.float32); ex[:, 0] = 1.; eI = np.repeat((ea + mm)[:, None], 2 ** N4, 1).astype(np.int32)
LB = np.maximum(tbox(P[:, 0], P[:, 2]), tbox(P[:, 1], P[:, 3])); far = LB > .5; ref = np.load('di4_ref_60_rho35.npy'); Q = ref[:, :4]; TR = ref[:, 6]
for SBL in (0, 1, 2):
    I_, X_, W_ = [ea], [eI], [ex]
    for a, b, c, own, ml, oth, hsa in ST: keep = np.where(own, oth, True if SBL == 0 else ml if SBL == 1 else (ml | ~hsa)); I_.append(a[keep]); X_.append(b[keep]); W_.append(c[keep])
    I = np.concatenate(I_); X = np.concatenate(X_); W = np.concatenate(W_); del I_, X_, W_; V, n, dt = gsolve(I, X, W, Vg, A.goal, N); ne = len(I); del I, X, W
    fin = V < BIG / 2; f = fin & far; r = V[f] / LB[f]; A.V = V; t0 = time.time(); T, sw = rollout(A, Q); tr = time.time() - t0; ok = np.isfinite(T); q = T[ok] / TR[ok]
    print('SBLAY %d | рёбер %.2fM | проходов %d, solve %.1f с | конечных %.3f | V/LB мед. %.3f mean %.3f p95 %.3f, V < .98 LB: %.3f | агент: дошли %d/60, T/эт(±.35) мед. %.3f mean %.3f p90 %.3f max %.2f, переключений мед. %d mean %.1f max %d, rollout %.0f с' % (SBL, ne / 1e6, n, dt, fin.mean(), np.median(r), r.mean(), np.percentile(r, 95), (r < .98).mean(), ok.sum(), np.median(q), q.mean(), np.percentile(q, 90), q.max(), np.median(sw[ok]), sw[ok].mean(), sw[ok].max(), tr), flush=True)
    np.save('V_sblay%d.npy' % SBL, V); np.save('T_sblay%d.npy' % SBL, T)
