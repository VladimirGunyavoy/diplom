"""research: хватает ли V в 2n+1 = 5 точках клетки, чтобы агент переключался по знаку dV/dv (DI, |u|<=1)?
V берём точный (T*), проверяем только представление: 5 точек -> линейный МНК -> градиент -> u = -sign(g_v).
Клетки: споры на сетке шага h (оба слоя), точки клетки: спора, спора +-r*n (поперёк поля), спора, унесённая на +-tau."""
import numpy as np, json, sys

def Ts(x, v):
    if x + v*abs(v)/2 > 0: return v + 2*np.sqrt(x + v*v/2)
    return -v + 2*np.sqrt(-x + v*v/2)

def flow(x, v, u, t): return x + v*t + u*t*t/2, v + u*t

def cell_points(s, u, r, tau):
    x, v = s; F = np.array([v, u]); n = np.array([-F[1], F[0]]) / np.linalg.norm(F)
    return np.array([s, s + r*n, s - r*n, flow(x, v, u, tau), flow(x, v, u, -tau)])

def build(h, r, tau, L=4.0):
    g = np.arange(-L, L + 1e-9, h); cells = []
    for x in g:
        for v in g:
            for u in (+1, -1):
                P = cell_points(np.array([x, v]), u, r, tau)
                V = np.array([Ts(*p) for p in P])
                A = np.c_[np.ones(5), P - P[0]]
                coef = np.linalg.lstsq(A, V, rcond=None)[0]
                cells.append((P[0], coef[1:]))
    C = np.array([c[0] for c in cells]); G = np.array([c[1] for c in cells])
    return C, G

def run(x0, v0, C, G, dt=0.01, tol=0.05):
    x, v, t, T0 = x0, v0, 0.0, Ts(x0, v0); nsw, u_prev = 0, 0
    while t < 3*T0 + 1:
        if np.hypot(x, v) < tol: return t, T0, nsw, True
        i = np.argmin((C[:, 0]-x)**2 + (C[:, 1]-v)**2)   # ближайшая спора (членство в клетке — не предмет теста)
        u = -np.sign(G[i, 1]) or 1.0
        nsw += (u != u_prev and u_prev != 0); u_prev = u
        x, v = flow(x, v, u, dt); t += dt
    return t, T0, nsw, False

rng = np.random.default_rng(0); starts = rng.uniform(-2, 2, (20, 2)); starts[0] = (-2, 0)
out = {}
for h in (0.4, 0.2, 0.1):
    C, G = build(h, r=h/2, tau=h/2)
    res = [run(x, v, C, G) for x, v in starts]
    ok = [r_ for r_ in res if r_[3]]
    rat = [r_[0]/r_[1] for r_ in ok]
    out[h] = dict(arrived=len(ok), n=len(res), ratio_mean=float(np.mean(rat)), ratio_max=float(np.max(rat)),
                  switches_mean=float(np.mean([r_[2] for r_ in ok])), first=res[0][:3])
    print(h, out[h]); sys.stdout.flush()
json.dump({str(k): v for k, v in out.items()}, open(__file__.replace('.py', '.json'), 'w'), indent=1, default=float)
