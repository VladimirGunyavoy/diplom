"""research hub-v5chain-research-9: споры-«бабочки» на дифдрайве (кинематика, состояние (x, y, θ), управление (v, ω) в ромбе |v| + |ω| ≤ 1, цель (0,0,0)).
Почему спора — точка (x, y) × окружность курсов: дуга с постоянным (v, ω) зависит только от (L = v·t, φ = ω·t) ⇒ из позы y в ЗАДАННУЮ точку (x_c, y_c)
ведут ровно две дуги (передом и задом), курс прихода θ_y + φ определён. Множество приходов — поверхность, курс в точке c ей трансверсален ⇒
«нормальный отрезок» споры = курс θ (весь круг, m узлов). Время дуги (ромб): t = |L| + |φ|. Плюс повороты на месте внутри споры.
Обратная задача — формулой: в теле позы y хорда (dx_b, dy_b), передом α = atan2(dy_b, dx_b), φ = 2α, L = ρ·α/sin α; задом α' = atan2(−dy_b, −dx_b),
φ = 2α', L = −ρ·α'/sin α'. Беллман: V(узел) = min[t + V_c(θ_прихода)] (V_c — линейно по θ, периодично), Якоби. Эталон — min(TGT, TGTGT) (`dd_rhombus_ref.py`)."""
import numpy as np, sys, time, json, os
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from dd_rhombus_ref import tgt, tgtgt
BIG = 1e3; LD = float(os.environ.get('LD', 3.)); RW = float(os.environ.get('RW', .6)); M = int(os.environ.get('M', 24))
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def arcs(Y, P):
    """дуги из поз Y (n,3) в точки P (n,2) (попарно): [(t, φ, L) передом, (t, φ, L) задом]."""
    dx, dy = P[:, 0] - Y[:, 0], P[:, 1] - Y[:, 1]; c, s = np.cos(Y[:, 2]), np.sin(Y[:, 2]); bx, by = c * dx + s * dy, -s * dx + c * dy; rho = np.hypot(bx, by); out = []
    for sg in (1., -1.):
        a = np.arctan2(sg * by, sg * bx); sinc = np.where(np.abs(a) > 1e-9, np.sin(a) / np.where(np.abs(a) > 1e-9, a, 1.), 1.)
        L = sg * rho / np.maximum(sinc, 1e-12); phi = 2 * a; out.append((np.abs(L) + np.abs(phi), phi, L))
    return out
def move(y, phi, L):
    x, yy, th = y
    if abs(phi) < 1e-9: return np.array([x + L * np.cos(th), yy + L * np.sin(th), th])
    R = L / phi; return np.array([x + R * (np.sin(th + phi) - np.sin(th)), yy - R * (np.cos(th + phi) - np.cos(th)), th + phi])
class ButterflyDD:
    def __init__(s, N=1500, m=M, seed=0):
        rng = np.random.default_rng(seed); s.C = np.r_[[[0., 0.]], rng.uniform(-LD, LD, (N, 2))]; s.K = len(s.C); s.m = m; s.th = 2 * np.pi * np.arange(m) / m
        s.V = np.full((s.K, m), BIG); s.V[0] = np.abs(wrap(s.th)); s.tree = cKDTree(s.C)                       # спора-цель: V = поворот на месте к 0
    def Vc(s, k, th):
        f = (th % (2 * np.pi)) / (2 * np.pi / s.m); j = np.floor(f).astype(int) % s.m; a = f - np.floor(f); j1 = (j + 1) % s.m
        V0, V1 = s.V[k, j], s.V[k, j1]; v = (1 - a) * V0 + a * V1; return np.where(((V0 >= BIG / 2) & (a < 1 - 1e-9)) | ((V1 >= BIG / 2) & (a > 1e-9)), BIG, v)
    def cand(s, Y, own=None):
        """для поз Y: все (iy, k, t, θ_прихода, φ, L) дуг к спорам в окне RW."""
        nb = s.tree.query_ball_point(Y[:, :2], RW); iy = np.repeat(np.arange(len(Y)), [len(a) for a in nb]); k = np.concatenate([np.asarray(a, int) for a in nb]) if len(iy) else np.zeros(0, int)
        if own is not None: g = k != own[iy]; iy, k = iy[g], k[g]
        R = arcs(Y[iy], s.C[k]); T = np.concatenate([r[0] for r in R]); PH = np.concatenate([r[1] for r in R]); L = np.concatenate([r[2] for r in R])
        iy2, k2 = np.r_[iy, iy], np.r_[k, k]; return iy2, k2, T, wrap(Y[iy2, 2] + PH), PH, L
    def solve(s, it=2000, tol=1e-7):
        Y = np.c_[np.repeat(s.C, s.m, 0), np.tile(s.th, s.K)]; own = np.repeat(np.arange(s.K), s.m)
        iy, k, T, thl, _, _ = s.cand(Y, own); s.npairs = len(T)
        f = (thl % (2 * np.pi)) / (2 * np.pi / s.m); j = np.floor(f).astype(int) % s.m; a = f - np.floor(f); j1 = (j + 1) % s.m
        o = np.argsort(iy, kind='stable'); iy, k, T, j, j1, a = iy[o], k[o], T[o], j[o], j1[o], a[o]; st = np.flatnonzero(np.r_[True, iy[1:] != iy[:-1]]); nodes = iy[st]
        dj = np.abs(wrap(s.th[None, :] - s.th[:, None]))                                                      # повороты на месте внутри споры
        V = s.V.reshape(-1).copy()
        for n in range(it):
            V0, V1 = V[k * s.m + j], V[k * s.m + j1]; val = T + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - 1e-9)) | ((V1 >= BIG / 2) & (a > 1e-9))] = BIG
            new = V.copy(); new[nodes] = np.minimum(V[nodes], np.minimum.reduceat(val, st))
            W = new.reshape(s.K, s.m); W = np.minimum(W, (W[:, None, :] + dj[None]).min(2)); new = W.reshape(-1); new[0] = 0.
            d = np.max(np.abs(np.minimum(new, BIG) - np.minimum(V, BIG))); V = new
            if d < tol: break
        s.V = V.reshape(s.K, s.m); s.n_it = n; return s
    def best(s, y, at=-1):
        """лучшее действие из позы y: дуга к споре (t, φ, L) или поворот на месте (если y в своей споре at)."""
        iy, k, T, thl, PH, L = s.cand(y[None], None if at < 0 else np.array([at])); J = T + s.Vc(k, thl) if len(T) else np.zeros(0); bJ, act = BIG, None
        if len(J): i = int(np.argmin(J)); bJ, act = J[i], ('arc', PH[i], L[i], int(k[i]))
        if at >= 0:
            dth = np.abs(wrap(s.th - y[2])); Jt = np.where(dth > 1e-9, dth + s.V[at], BIG); j = int(np.argmin(Jt))                   # поворот «на 0» запрещён (иначе ничья = вечный цикл)
            if Jt[j] < bJ - 1e-9: bJ, act = Jt[j], ('turn', wrap(s.th[j] - y[2]), 0., at)
        if np.hypot(y[0], y[1]) < 1e-9: Jg = abs(wrap(y[2])); bJ, act = (Jg, ('turn', -wrap(y[2]), 0., 0)) if Jg <= bJ else (bJ, act)
        return bJ, act
    def rollout(s, y0, smax=300):
        y = np.array(y0, float); t = 0.; at = -1; segs = 0
        for _ in range(smax):
            if np.hypot(y[0], y[1]) < 1e-9 and abs(wrap(y[2])) < 1e-9: return t, segs
            J, act = s.best(y, at)
            if act is None or J >= BIG / 2: return np.inf, segs
            kind, phi, L, k = act
            if kind == 'turn': y = np.array([y[0], y[1], y[2] + phi]); t += abs(phi)
            else: y = move(y, phi, L); y[:2] = s.C[k]; t += abs(L) + abs(phi)                                 # конец дуги = точка споры (точно, формула)
            at = k; segs += 1
        return np.inf, segs

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; NQ = int(os.environ.get('NQ', 60))
    rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (NQ, 2)), rng.uniform(-np.pi, np.pi, NQ)]
    rf = 'dd_ref_%d.npy' % NQ
    if os.path.exists(rf): ref = np.load(rf)
    else: t0 = time.time(); ref = np.array([min(tgt(*q), tgtgt(*q)) for q in Q]); np.save(rf, ref); print('эталон', round(time.time() - t0), 'с', flush=True)
    t0 = time.time(); B = ButterflyDD(N=N).solve(); print(json.dumps(dict(N=N, m=B.m, RW=RW, nodes=B.K * B.m, pairs=B.npairs, iters=B.n_it, sec=round(time.time() - t0), BIG=round(float((B.V >= BIG / 2).mean()), 3))), flush=True)
    V = np.array([B.best(q)[0] for q in Q]); R = [B.rollout(q) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T)
    def st(r): return dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 3), max=round(float(r.max()), 3)) if len(r) else None
    print(json.dumps(dict(V_over_ref=st(V[V < BIG / 2] / ref[V < BIG / 2]), reach=round(float(f.mean()), 3), T_over_ref=st(T[f] / ref[f]), segs_med=float(np.median(S[f])) if f.any() else None, sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save('butterfly_dd_N%d_m%d_rw%g.npy' % (N, B.m, RW), np.array([ref, V, T, S]))
