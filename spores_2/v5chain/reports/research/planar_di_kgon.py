"""research: плоский DI ẍ = a, |a|₂ ≤ 1, слои = K направлений (вписанный K-угольник). Потеря времени против круга (K = 64 ≈ круг).
Минимальное время: для фиксированного T — ЛП на выполнимость (N шагов, a_t = Σ_k λ_{t,k}·e_k, λ ≥ 0, Σλ ≤ 1 — выпуклая оболочка
K-угольника с нулём, т.е. допускает дребезг между соседними вершинами), бисекция по T. Точная дискретизация кусочно-постоянного a."""
import numpy as np, json
from scipy.optimize import linprog
def feasible(T, p0, v0, K, N=40):
    dt = T / N; E = np.stack([np.cos(2*np.pi*np.arange(K)/K), np.sin(2*np.pi*np.arange(K)/K)], 1)
    # конец: p_N = p0 + v0 T + Σ_t a_t·(dt²/2 + dt·(T − (t+1)dt)), v_N = v0 + Σ_t a_t dt
    wp = dt*dt/2 + dt*(T - dt*np.arange(1, N+1)); wv = np.full(N, dt)
    Aeq = np.zeros((4, N*K)); beq = np.zeros(4)
    for t in range(N):
        for k in range(K):
            j = t*K + k; Aeq[0:2, j] = wp[t]*E[k]; Aeq[2:4, j] = wv[t]*E[k]
    beq[0:2] = -(p0 + v0*T); beq[2:4] = -v0
    Aub = np.zeros((N, N*K)); [Aub.__setitem__((t, slice(t*K, (t+1)*K)), 1) for t in range(N)]
    r = linprog(np.zeros(N*K), A_ub=Aub, b_ub=np.ones(N), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    return r.status == 0
def tmin(p0, v0, K, lo=0.0, hi=20.0, it=30):
    for _ in range(it):
        m = (lo + hi)/2; lo, hi = (lo, m) if feasible(m, p0, v0, K) else (m, hi)
    return hi
rng = np.random.default_rng(0); S = [(rng.uniform(-2, 2, 2), rng.uniform(-1, 1, 2)) for _ in range(12)]
ref = [tmin(p, v, 64) for p, v in S]; out = {}
for K in (4, 6, 8, 12):
    rat = np.array([tmin(p, v, K) for p, v in S]) / np.array(ref)
    bound = 1/np.sqrt(np.cos(np.pi/K))
    out[K] = dict(mean=float(rat.mean()), max=float(rat.max()), bound_rest=float(bound))
    print(f"K={K}: T_K/T_круг mean {rat.mean():.4f} max {rat.max():.4f}  (оценка худшего 1/√cos(π/K) = {bound:.4f})", flush=True)
json.dump({str(k): v for k, v in out.items()}, open(__file__.replace('.py', '.json'), 'w'), indent=1)
