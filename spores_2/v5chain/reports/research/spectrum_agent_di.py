"""research hub-research-6: агент по слоям (вершины U) vs агент по спектру (непрерывный u из интерполяции образа U), DI, V = T* точно.
Шаг τ: u = argmin V(Z(u)), Z(u) — образ U за τ по 3 слоям (−1, 0, +1) (для DI интерполяция точна). Цель: |x|,|v| ≤ ε. Метрики: T/T*, переключений, TV(u)."""
import numpy as np, sys

def Tstar(x, v):
    s = x + v*abs(v)/2
    if s > 0: return v + 2*np.sqrt(v*v/2 + x)
    if s < 0: return -v + 2*np.sqrt(v*v/2 - x)
    return abs(v)
def phi(p, u, t): x, v = p; return np.array([x + v*t + u*t*t/2, v + u*t])

def run(p0, tau, mode, eps=0.02, Tmax=40):
    p = np.array(p0, float); t = 0.0; us = []
    lay = {-1: phi(p, -1, tau), 0: phi(p, 0, tau), 1: phi(p, 1, tau)}
    while t < Tmax:
        if max(abs(p)) <= eps: return t, us
        L = {k: phi(p, k, tau) for k in (-1, 0, 1)}
        Z = lambda u: L[0] + u*(L[1]-L[-1])/2 + u*u*((L[1]+L[-1])/2 - L[0])
        if mode == 'bb2': cand = [-1., 1.]
        elif mode == 'bb3': cand = [-1., 0., 1.]
        else:
            g = np.linspace(-1, 1, 201); cand = list(g)
        vals = [Tstar(*Z(u)) for u in cand]; u = cand[int(np.argmin(vals))]
        if mode == 'spec':                                  # уточнение золотым сечением около лучшего узла
            a, b = max(-1, u - .01), min(1, u + .01)
            for _ in range(30):
                c, d = b - .618*(b-a), a + .618*(b-a)
                if Tstar(*Z(c)) < Tstar(*Z(d)): b = d
                else: a = c
            u = (a+b)/2
        # последний шаг: если цель достижима внутри шага — укоротить шаг (честно: агент знает модель)
        p = Z(u); t += tau; us.append(u)
    return np.inf, us

rng = np.random.default_rng(0); P0 = rng.uniform(-3, 3, (40, 2))
for tau in (.1, .25, .5):
    for mode in ('bb2', 'bb3', 'spec'):
        R = []; sw = []; tv = []; fail = 0
        for p0 in P0:
            T, us = run(p0, tau, mode)
            if not np.isfinite(T): fail += 1; continue
            R.append(T/Tstar(*p0)); us = np.array(us); sw.append(np.sum(np.abs(np.diff(np.sign(np.round(us, 6)))) > 0)); tv.append(np.sum(np.abs(np.diff(us))))
        R = np.array(R)
        print(f'τ={tau:<4} {mode:5} успех {40-fail}/40  T/T* mean {R.mean():.3f} max {R.max():.3f}  смен знака med {np.median(sw):.0f}  TV(u) med {np.median(tv):.1f}')

# --- (2) свободное время шага t ∈ (0, τ]: критерий t + V(φ_t(x,u)) (образ U × [0,τ] — (m+1)-мерный) ---
def run_t(p0, tau, mode, eps=0.02, Tmax=40):
    p = np.array(p0, float); t = 0.0; us = []
    cand = [-1., 1.] if mode == 'bb_t' else list(np.linspace(-1, 1, 81))
    ts = np.linspace(tau/40, tau, 40)
    while t < Tmax:
        if max(abs(p)) <= eps: return t, us
        best = min(((s + Tstar(*phi(p, u, s)), u, s) for u in cand for s in ts))
        _, u, s = best; p = phi(p, u, s); t += s; us.append(u)
    return np.inf, us
if 'free' in sys.argv:
    for tau in (.1, .25, .5, 1.0):
        for mode in ('bb_t', 'spec_t'):
            R = []; n = []; fail = 0
            for p0 in P0:
                T, us = run_t(p0, tau, mode)
                if not np.isfinite(T): fail += 1; continue
                R.append(T/Tstar(*p0)); n.append(len(us))
            R = np.array(R); print(f'τ={tau:<4} {mode:6} успех {40-fail}/40  T/T* mean {R.mean():.3f} max {R.max():.3f}  шагов med {np.median(n):.0f}')
