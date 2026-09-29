"""research: точки поперёк клетки по гессиану vs равномерно (DI, V = T*), одинаковое число точек N.
Сечение: отрезок длины 2r через спору перпендикулярно полю u=+1. Кусочно-линейная интерполяция V по N точкам, ошибка на 2001 точке.
Плотность по гессиану: точки — равнораспределение ρ(s) = (|V''(s)| + eps)^(1/2) (оптимум для L∞ линейной интерполяции),
V'' — оракул (конечные разности на мелкой сетке). eps = 1e-3·max|V''| (иначе у излома все точки слипаются)."""
import numpy as np, json
from di_gradV_switch import Ts

def section(s, r, m=2001):
    x, v = s; F = np.array([v, 1.0]); n = np.array([-1.0, v]) / np.linalg.norm(F)
    p = np.linspace(-r, r, m); P = np.array(s)[None] + p[:, None]*n[None]
    return p, np.array([Ts(*q) for q in P])

def interp_err(p, V, idx):
    return np.max(np.abs(np.interp(p, p[idx], V[idx]) - V))

def hess_idx(p, V, N, eps_rel=1e-3):
    d2 = np.abs(np.gradient(np.gradient(V, p), p)); rho = np.sqrt(d2 + eps_rel*d2.max() + 1e-12)
    cdf = np.concatenate([[0], np.cumsum(0.5*(rho[1:]+rho[:-1])*np.diff(p))]); cdf /= cdf[-1]
    return np.unique(np.searchsorted(cdf, np.linspace(0, 1, N)).clip(0, len(p)-1))

rng = np.random.default_rng(1); spores = rng.uniform(-2, 2, (200, 2)); out = {}
for r in (0.2, 0.5):
    for N in (3, 5, 9):
        eu, eh = [], []
        for s in spores:
            p, V = section(s, r)
            eu.append(interp_err(p, V, np.linspace(0, len(p)-1, N).astype(int)))
            eh.append(interp_err(p, V, hess_idx(p, V, N)))
        eu, eh = np.array(eu), np.array(eh); k = f"r={r} N={N}"
        out[k] = dict(uni_mean=eu.mean(), uni_max=eu.max(), hes_mean=eh.mean(), hes_max=eh.max(),
                      hes_better_frac=float(np.mean(eh < eu*0.999)), hes_worse_frac=float(np.mean(eh > eu*1.001)))
        print(k, {a: round(b, 4) for a, b in out[k].items()}, flush=True)
json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1, default=float)
