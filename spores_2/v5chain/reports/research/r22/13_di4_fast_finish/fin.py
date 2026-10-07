"""research-22, эксп. 13: быстрый финиш у цели вместо finish_gen.shoot (52 топологии × SLSQP × rk4 на Python, 5.8 с/запрос).
shoot_grid: полный перебор времён ≤ NA дуг на сетке шага H пачкой rk4 (все топологии и времена сразу), минимум суммарного времени среди концов в коробке .9·RHOV.
Проверка: штатный growN.rollout, VF как у эксп. 12в, di4 L1600, V SBLAY 1; эталон ±.35."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
H = float(os.environ.get('FH', .05)); NCALL = [0, 0.]
def shoot_grid(y, f, US, RHOV, wrapy, tmax, NA=3, inits=2):
    t0 = time.time(); K = int(round(tmax / H)); R = .9 * np.asarray(RHOV); best = (np.inf, None); nu = len(US)
    def traj(Y, u):
        out = [Y]
        for _ in range(K): Y = G.rk4(Y, u, H, 1); out.append(Y)
        return np.stack(out, 1)                                                                # (B, K + 1, N), время j·H
    def upd(S, T, tp):
        nonlocal best
        ok = (np.abs(wrapy(S)) <= R).all(-1)
        if ok.any():
            t = np.where(ok, T, np.inf).min()
            if t < best[0]: best = (float(t), tp)
    tj = np.arange(K + 1) * H; L1 = {a: traj(np.asarray(y, float)[None], US[a]) for a in range(nu)}
    for a in range(nu): upd(L1[a], tj[None], (a,))
    if NA >= 2:
        for a in range(nu):
            S1 = L1[a][0, 1:]; T1 = tj[1:]
            for b in range(nu):
                if b == a: continue
                S2 = traj(S1, US[b]); T2 = T1[:, None] + tj[None]; upd(S2, T2, (a, b))
                if NA >= 3:
                    m = T2[:, 1:] < min(best[0], tmax); S2f = S2[:, 1:][m]; T2f = T2[:, 1:][m]
                    for c in range(nu):
                        if c == b or not len(S2f): continue
                        S3 = traj(S2f, US[c]); upd(S3, T2f[:, None] + tj[None], (a, b, c))
    NCALL[0] += 1; NCALL[1] += time.time() - t0; return best
if __name__ == '__main__':
    d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); r = np.load('di4_ref_60_rho35.npy'); Q = r[:, :4]; TR = r[:, 6]; A.V = np.load(os.environ['VFILE'])
    stop = np.c_[Q[:, 0] + Q[:, 2] * abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * abs(Q[:, 3]) / 2]; inf_ = np.abs(stop).max(1) <= 2.5; A.vstar(Q)
    if int(os.environ.get('GRID', 1)): G.FG.shoot = shoot_grid
    t0 = time.time(); T, sw, path = A.rollout(Q); dt = time.time() - t0; ok = np.isfinite(T); q = T / TR; m = ok & inf_
    print('VF %s %s H %.3f | дошли %d/60 (в поле %d/%d) | T/эт в поле: мед. %.3f mean %.3f p90 %.3f max %.2f | T < эталона у %d | переключений мед. %d max %d | %.1f с на 60 (%.0f мс/запрос) | стрельб %d, %.0f мс каждая' % (os.environ.get('VF', '0'), 'сетка' if int(os.environ.get('GRID', 1)) else 'SLSQP', H, ok.sum(), m.sum(), inf_.sum(), np.median(q[m]), q[m].mean(), np.percentile(q[m], 90), q[m].max(), (T[m] < TR[m] - 1e-6).sum(), np.median(sw[m]), sw[m].max(), dt, 1e3 * dt / 60, NCALL[0], 1e3 * NCALL[1] / max(NCALL[0], 1)), flush=True)
