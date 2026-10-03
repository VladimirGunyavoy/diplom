"""research hub-v5chain-research-9: бабочки дифдрайва + 2 диска-препятствия (TASK: «дифдрайв с препятствиями (объезд)»). Как butterfly_dd.py,
но споры внутри дисков убраны, дуга годна, только если 16 точек вдоль неё вне дисков (робот — точка). Эталона в замкнутой форме нет:
сходимость по числу спор (N → 2N) + проверка, что без дисков код даёт прежние цифры. Финиш min(TGT, TGTGT) — только если путь финиша вне дисков."""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
import butterfly_dd as M
DISKS = np.array([[1.0, 0.3, .45], [-0.8, -0.9, .4]]) if not os.environ.get('NODISK') else np.zeros((0, 3))
def move_v(Y, phi, L):
    x, y, th = Y[:, 0], Y[:, 1], Y[:, 2]; st = np.abs(phi) < 1e-9; R = np.where(st, 0., L / np.where(st, 1., phi))
    return np.c_[np.where(st, x + L * np.cos(th), x + R * (np.sin(th + phi) - np.sin(th))), np.where(st, y + L * np.sin(th), y - R * (np.cos(th + phi) - np.cos(th)))]
def free(P): return np.all([np.hypot(P[:, 0] - d[0], P[:, 1] - d[1]) > d[2] for d in DISKS], 0) if len(DISKS) else np.ones(len(P), bool)
class ObstDD(M.ButterflyDD):
    def __init__(s, N, **kw):
        super().__init__(N=N, **kw); keep = free(s.C); keep[0] = True; s.C = s.C[keep]; s.K = len(s.C); s.V = s.V[keep]; s.tree = M.cKDTree(s.C)
    def cand(s, Y, own=None):
        iy, k, T, thl, PH, L = super().cand(Y, own); ok = np.ones(len(T), bool)
        for fr in np.linspace(0, 1, 17)[1:-1]: ok &= free(move_v(Y[iy], PH * fr, L * fr))
        return iy[ok], k[ok], T[ok], thl[ok], PH[ok], L[ok]
if __name__ == '__main__':
    N = int(sys.argv[1]); rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (120, 2)), rng.uniform(-np.pi, np.pi, 120)]; Q = Q[free(Q)][:60]
    t0 = time.time(); B = ObstDD(N).solve(); print(json.dumps(dict(N=N, RW=M.RW, disks=len(DISKS), spores=B.K, pairs=B.npairs, iters=B.n_it, sec=round(time.time() - t0))), flush=True)
    V = np.array([B.best(q)[0] for q in Q]); R = [B.rollout(q) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T)
    print(json.dumps(dict(reach=round(float(f.mean()), 3), T_over_V=round(float(np.mean(T[f] / V[f])), 4), T_mean=round(float(T[f].mean()), 4), segs_med=float(np.median(S[f])), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save('butterfly_dd_obst_N%d_rw%g%s.npy' % (N, M.RW, '_nodisk' if not len(DISKS) else ''), np.array([V, T, S]))
