"""research hub-research-6: «слепая» допустимая эвристика (PLAN 0з п.6): T ≥ h(x) = max(T*_DI(Δθ, ω; a_max), (|Δθ|−R)/WM), a_max = g/l + u_max ≥ |θ̈|.
Маятник u=.3/.5, цель верх (π,0), окно R. Сравнение с V по сетке (PendAtlas v6, τ .1). Метрики: доля h/V (допустимость: h ≤ V(1+погр. сетки)), медиана h/V."""
import sys, numpy as np; sys.path.insert(0, '.')
from src.atlas6.pend import PendAtlas
def TDI(x, v, a):                                    # быстродействие DI до (0,0), |u|≤a: масштаб x/a
    x = x / a; v = v / a; s = x + v*np.abs(v)/2
    return np.where(s > 0, v + 2*np.sqrt(np.maximum(v*v/2 + x, 0)), np.where(s < 0, -v + 2*np.sqrt(np.maximum(v*v/2 - x, 0)), np.abs(v)))
for um in (.3, .5):
    P = PendAtlas(n_th=144, n_w=121, wmax=4.0, tau=0.1, umax=um, R_goal=0.3); it = P.solve(iters=3000)
    rng = np.random.default_rng(0); X = np.stack([rng.uniform(-np.pi, np.pi, 4000), rng.uniform(-3, 3, 4000)], -1)
    V = P.interp(P.V, X); ok = (V < 100) & ~P.in_goal(X.T, 0.3)
    dth = (X[:, 0] - np.pi + np.pi) % (2*np.pi) - np.pi                       # кратчайшее Δθ (обе стороны круга)
    h = np.maximum(np.minimum(TDI(dth, X[:, 1], 1 + um), TDI(dth - 2*np.pi*np.sign(dth), X[:, 1], 1 + um)), np.maximum(np.abs(dth) - .3, 0)/4.0)
    h = np.maximum(h - 0.3/np.sqrt(1+um)*1.5, 0)                               # окно цели R: грубая поправка вниз (допустимость)
    r = h[ok]/V[ok]
    bad = np.where(r > 1)[0]; Xo, Vo, ho = X[ok], V[ok], h[ok]; print("  нарушения (θ, ω, V, h):", [tuple(np.round([*Xo[i], Vo[i], ho[i]], 2)) for i in bad[:6]])
    print(f'u={um}: итераций {it}, точек {ok.sum()}, h/V медиана {np.median(r):.2f}, 10% {np.quantile(r,.1):.2f}, 90% {np.quantile(r,.9):.2f}, h>V: {np.mean(r>1.0):.3f} (макс {r.max():.2f}); V медиана {np.median(V[ok]):.1f}', flush=True)
