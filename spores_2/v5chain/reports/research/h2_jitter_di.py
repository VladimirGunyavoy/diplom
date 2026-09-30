"""research H2 (adaptive_vs_grid.md): убирают ли цепочки спор накопление ошибки V? Тот же набор спор (адаптив worker-4, m=8), тот же
решатель (solve_scattered, Делоне), но споры сдвинуты случайно на σ (в долях τ·|F|~шага цепочки) — выравнивание «выход = следующая спора»
ломается при том же N и том же распределении. Если V/T* заметно растёт с σ — H2 подтверждена."""
import sys, numpy as np
sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v6')
from src.atlas6.adaptive_di import spores_for, solve_scattered, V_query
from src.atlas6.agent import T_star
rng = np.random.default_rng(0); Q = [rng.uniform(-3, 3, 2) for _ in range(10)]; tau = 0.25
for sig in (0.0, 0.01, 0.03, 0.1):
    r, Ns = [], []
    for x0 in Q:
        P = spores_for(x0, tau, 1.3*T_star(*x0) + 1, 8)
        keep = np.linalg.norm(P - x0, axis=1) > 1e-9                        # старт и цель не двигать
        keep &= np.linalg.norm(P, axis=1) > 1e-9
        P = P + np.where(keep[:, None], rng.normal(0, sig, P.shape), 0.0)
        tri, V, _ = solve_scattered(P, tau); Ns.append(len(P)); r.append(V_query(P, V, tri, x0) / T_star(*x0))
    r = np.array(r); print(f"σ={sig:.2f}: N≈{np.mean(Ns):.0f}  V/T* mean {r.mean():.3f} max {r.max():.3f} min {r.min():.3f}", flush=True)
