"""research-22, эксп. 11: гистерезис переключений агента на di4 L400 — держать текущее u, пока лучшее u не выигрывает больше EPS (петли: T/LB mean 2.3 при мед. 1.21).
Тот же rollout growN (VF 0), V без защёлки PESS 1; 60 стартов, T к точной LB в коробку ±.35 (`../di4_ref_60_rho35.npy`)."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
DTN = G.DTN; US = G.US
def rollout(s, Q, eps, tmax=40.):
    Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); sw = np.zeros(n, int); pk = np.full(n, -1); ar = np.arange(n)
    for _ in range(int(tmax / DTN)):
        if done.all(): break
        tgs = np.stack([s.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([s.vstar(G.step(Y, u)) for u in US], 1)); k = J.argmin(1)
        hold = (pk >= 0) & (J[ar, np.maximum(pk, 0)] <= J[ar, k] + eps); k = np.where(hold, pk, k); tg = tgs[ar, k]; stuck = J[ar, k] >= G.BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
        for ki, ui in enumerate(US):
            m = act & (k == ki)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        sw += act & (pk >= 0) & (k != pk); pk = np.where(act, k, pk); Y = G.wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= G.ingoal(Y) | stuck
    T[~G.ingoal(Y)] = np.inf; return T, sw
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = np.load(os.environ['VF_']); ref = np.load('di4_ref_60_rho35.npy'); Q = ref[:, :4]; LB = ref[:, 6]; V0 = A.vstar(Q)
print('V(старт) конечна у %d/60' % (V0 < G.BIG / 2).sum(), flush=True)
for eps in (0., .01, .03, .06, .1, .2, .4):
    t0 = time.time(); T, sw = rollout(A, Q, eps); ok = np.isfinite(T); r = T[ok] / LB[ok]
    print('EPS %.2f | дошли %d/60 | T/LB mean %.3f мед. %.3f p90 %.3f max %.2f | переключений mean %.1f мед. %d max %d | %.0f с' % (eps, ok.sum(), r.mean(), np.median(r), np.percentile(r, 90), r.max(), sw[ok].mean(), np.median(sw[ok]), sw[ok].max(), time.time() - t0), flush=True)
