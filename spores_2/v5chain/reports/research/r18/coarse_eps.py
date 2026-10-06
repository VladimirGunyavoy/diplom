# Какой запас ε нужен, если V̂, T̂_s — с ГРУБОГО атласа (модель: мультилинейная интерполяция точных значений по сетке G^4 на [-2.5, 2.5]^4).
# Оптимальная траектория DI 4D: медленная ось — bang-bang, быстрая — оптимум и покой в 0 (тоже оптимальна). ε_need = max по точкам пути (T̂_s + V̂)/Ĉ − 1.
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from region_frac_lib import t2
def V4(P): return np.maximum(t2(P[..., 0], P[..., 1], 0., 0.), t2(P[..., 2], P[..., 3], 0., 0.))
def T4(s, P): return np.maximum(t2(s[0], s[1], P[..., 0], P[..., 1]), t2(s[2], s[3], P[..., 2], P[..., 3]))
def axis_traj(x, v, t):
    """оптимальный DI одной оси в (0,0), потом покой; положение на временах t"""
    T = t2(np.array(x), np.array(v), 0., 0.); out = []
    dt = 1e-3; y = np.array([x, v], float); tt = 0.; rec = {}; ts = sorted(t); j = 0
    while j < len(ts):
        while j < len(ts) and ts[j] <= tt + 1e-12: rec[ts[j]] = y.copy(); j += 1
        if abs(y[0]) < 1e-9 and abs(y[1]) < 1e-9 or tt >= T: u = 0.; y = np.zeros(2) if tt >= T else y
        else:
            sw = y[0] + y[1] * abs(y[1]) / 2; u = -np.sign(sw) if abs(sw) > 1e-9 else -np.sign(y[1])
        y = y + dt * np.array([y[1] + u * dt / 2, u]); tt += dt
    return np.array([rec[q] for q in t])
rng = np.random.default_rng(3); M = 200000; P = rng.uniform(-2.5, 2.5, (M, 4)); Vp = V4(P)
for G in (5, 7, 9):
    g = np.linspace(-2.5, 2.5, G); GG = np.stack(np.meshgrid(g, g, g, g, indexing='ij'), -1)
    Vh = RGI((g,) * 4, V4(GG), bounds_error=False, fill_value=None); need = []; fr = []
    for j in range(20):
        s = rng.uniform(-2, 2, 4); C = V4(s); t = np.linspace(0, C, 60)
        path = np.concatenate([axis_traj(s[0], s[1], t), axis_traj(s[2], s[3], t)], 1)
        Th = RGI((g,) * 4, T4(s, GG), bounds_error=False, fill_value=None); Ch = Vh(s[None])[0]
        e = np.max((Th(path) + Vh(path)) / Ch) - 1; need.append(e)
        fr.append(np.mean(Th(P) + Vh(P) <= (1 + max(e, 0) + .05) * Ch))
    need = np.array(need)
    print(f"G {G}: шаг {5/(G-1):.2f} | ε_need мед {np.median(need):.3f} 90% {np.quantile(need,.9):.3f} max {need.max():.3f} | доля поля при ε_need+.05: мед {np.median(fr):.2e} max {np.max(fr):.2e}", flush=True)
