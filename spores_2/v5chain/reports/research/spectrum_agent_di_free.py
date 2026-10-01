"""research hub-research-6: DI, агент со свободным временем шага t ∈ (0, τ]: (u,t) = argmin t + T*(φ_t(x,u)). bb_t — u ∈ {±1}, spec_t — u ∈ [−1,1] (81 узел).
Образ U × [0,τ] — (m+1)-мерный (для DI 2D = размерность состояния ⇒ точное попадание в цель за шаг, когда цель внутри)."""
import numpy as np
def Ts(x, v):
    s = x + v*np.abs(v)/2
    return np.where(s > 0, v + 2*np.sqrt(np.maximum(v*v/2 + x, 0)), np.where(s < 0, -v + 2*np.sqrt(np.maximum(v*v/2 - x, 0)), np.abs(v)))
def run(p0, tau, mode, eps=0.02, Tmax=40):
    U = np.array([-1., 1.]) if mode == 'bb_t' else np.linspace(-1, 1, 81); S = np.linspace(tau/40, tau, 40)
    UU, SS = np.meshgrid(U, S); UU, SS = UU.ravel(), SS.ravel(); x, v = p0; t = 0; n = 0
    while t < Tmax:
        if max(abs(x), abs(v)) <= eps: return t, n
        xn = x + v*SS + UU*SS**2/2; vn = v + UU*SS; k = np.argmin(SS + Ts(xn, vn))
        x, v, t, n = xn[k], vn[k], t + SS[k], n + 1
    return np.inf, n
rng = np.random.default_rng(0); P0 = rng.uniform(-3, 3, (40, 2))
for tau in (.1, .25, .5, 1.0):
    for mode in ('bb_t', 'spec_t'):
        r = np.array([run(p, tau, mode) for p in P0]); ok = np.isfinite(r[:, 0]); R = r[ok, 0]/Ts(P0[ok, 0], P0[ok, 1])
        print(f'τ={tau:<4} {mode:6} успех {ok.sum()}/40  T/T* mean {R.mean():.3f} max {R.max():.3f}  шагов med {np.median(r[ok,1]):.0f}', flush=True)
