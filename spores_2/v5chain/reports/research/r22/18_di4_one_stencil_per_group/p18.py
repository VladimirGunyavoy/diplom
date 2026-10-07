"""research-22, эксп. 18: ОДИН стенсил на группу (узел, u, слой приземления) — самый центральный (min наибольшего веса) — вместо min/среднего по перекрывающимся.
Зачем: на полных слоях (Lfull, 25 650 клеток, 9.46M узлов) пар 15.9 на узел и u (600M) — не помещаются ни в RAM 60 ГБ, ни в GPU 16 ГБ; так рёбер ≈ 4–5 на узел при любом перекрытии.
Стенсилы строятся потоково (пачки узлов), фильтр SBLAY 1 и выбор — сразу. Функции — из эксп. 12: di4 с полными слоями (L1600 от b5, п.38) — весь конвейер после build на GPU: стенсилы (sgpu.test_gpu) → рёбра SBCAUS при SBLAY 0/1/2 →
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

CHN = int(os.environ.get('CHN', 400000)); t0 = time.time(); d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); N = A.N; P = A.P; tag = os.environ.get('TAG', 'x')
print('[%s] слои %s клеток, узлов %d' % (tag, [len(l) for l in A.layers], N), flush=True)
cl = np.concatenate([[k] * len(l) for k, l in enumerate(A.layers)]); O = np.array([c.o for c in A.cells]); nl = cl[np.searchsorted(O, np.arange(N), 'right') - 1]; Vg = np.full(N, np.inf); I_, X_, W_ = [], [], []; npair = 0; nkept = 0; t0 = time.time()
for ui, u in enumerate(US):
    Vg = np.minimum(Vg, A.tgoal(P, u)); cov = np.zeros(N, bool)
    for s0 in tqdm(range(0, N, CHN), desc='стенсилы u %d' % ui, mininterval=10):
        a, b, c = A.stencils(G.step(P[s0:s0 + CHN], u)); a = a + s0; npair += len(a); nc = np.searchsorted(O, a, 'right') - 1; vc = np.searchsorted(O, b[:, 0], 'right') - 1; own = cl[nc] == ui; dl = cl[vc]
        keep = np.where(own, vc != nc, dl == ui); a, b, c, dl = a[keep], b[keep], c[keep], dl[keep]; cov[a[nl[a] != ui]] = True
        key = a.astype(np.int64) * 4 + dl; o = np.lexsort((c.max(1), key)); key = key[o]; f = o[np.r_[True, key[1:] != key[:-1]]] if len(o) else o
        I_.append(a[f].astype(np.int32)); X_.append(b[f].astype(np.int32)); W_.append(c[f].astype(np.float32)); nkept += len(f)
    print('[%s] u %d: узлов чужих слоёв со стенсилом в слой u: %.3f' % (tag, ui, cov[nl != ui].mean()), flush=True)
ts = time.time() - t0; mm = G.M ** (N4 - 1); ea = np.concatenate([c.o + np.arange((c.G.shape[0] - 1) * mm) for c in A.cells]).astype(np.int32); ex = np.zeros((len(ea), 2 ** N4), np.float32); ex[:, 0] = 1.; eI = np.repeat((ea + mm)[:, None], 2 ** N4, 1).astype(np.int32)
I = np.concatenate(I_ + [ea]); X = np.concatenate(X_ + [eI]); W = np.concatenate(W_ + [ex]); del I_, X_, W_
print('[%s] стенсилы 4 u (GPU, потоково): %.0f с | пар всего %.1fM (%.2f на узел и u) → оставлено %.1fM + точных %.1fM = рёбер %.1fM (%.2f на узел)' % (tag, ts, npair / 1e6, npair / N / 4, nkept / 1e6, len(ea) / 1e6, len(I) / 1e6, len(I) / N), flush=True)
V, n, dt = gsolve(I, X, W, Vg, A.goal, N); del I, X, W; LB = np.maximum(tbox(P[:, 0], P[:, 2]), tbox(P[:, 1], P[:, 3])); far = LB > .5; fin_ = V < BIG / 2; f = fin_ & far; r = V[f] / LB[f]
ref = np.load('di4_ref_60_rho35.npy'); Q = ref[:, :4]; TR = ref[:, 6]; inf_ = np.abs(np.c_[Q[:, 0] + Q[:, 2] * abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * abs(Q[:, 3]) / 2]).max(1) <= 2.5; A.V = V; np.save('V_%s.npy' % tag, V)
T, sw = rollout(A, Q); ok = np.isfinite(T); m = ok & inf_; q = T / TR
print('[%s] ОДИН стенсил на группу, SBLAY 1 | проходов %d, solve %.1f с | конечных %.3f | V/LB мед. %.3f mean %.3f, V < .98 LB: %.3f, V < LB − .1: %.3f | агент без финиша: дошли %d/60 (в поле %d/%d), T/эт в поле мед. %.3f mean %.3f p90 %.3f max %.2f' % (tag, n, dt, fin_.mean(), np.median(r), r.mean(), (r < .98).mean(), ((V - LB)[f] < -.1).mean(), ok.sum(), m.sum(), inf_.sum(), np.median(q[m]), q[m].mean(), np.percentile(q[m], 90), q[m].max()), flush=True)
import fin2 as F2
G.FG.shoot = F2.shoot_pol; G.VF = 1.; t0 = time.time(); T, sw, path = A.rollout(Q); dt = time.time() - t0; ok = np.isfinite(T); m = ok & inf_; q = T / TR
print('[%s] + финиш shoot_pol VF 1 | в поле %d/%d | T/эт мед. %.3f mean %.3f p90 %.3f max %.2f | %.0f мс/запрос' % (tag, m.sum(), inf_.sum(), np.median(q[m]), q[m].mean(), np.percentile(q[m], 90), q[m].max(), 1e3 * dt / 60), flush=True)
