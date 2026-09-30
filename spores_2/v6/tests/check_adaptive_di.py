"""H1 (research/adaptive_vs_grid.md), DI: V(старт)/T* — адаптивные споры под запрос против решётки; ОДИН решатель (solve_scattered, Делоне-интерполяция), τ=.25. Запуск из v6, ~30 с."""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_di import *
rng = np.random.default_rng(0); Q = [rng.uniform(-3, 3, 2) for _ in range(10)]; tau = 0.25
for h in (1.0, 0.5, 0.25):
    xs = np.arange(-6, 6 + 1e-9, h); P = np.array([(x, v) for x in xs for v in xs]); tri, V, it = solve_scattered(P, tau)
    r = [V_query(P, V, tri, x0) / T_star(*x0) for x0 in Q]; print('решётка h=%.2f N=%d: V/T* mean %.3f max %.3f' % (h, len(P), np.mean(r), np.max(r)), flush=True)
for m in (8, 4, 2):
    r = []; Ns = []
    for x0 in Q:
        P = spores_for(x0, tau, 1.3 * T_star(*x0) + 1, m); tri, V, it = solve_scattered(P, tau); Ns.append(len(P)); r.append(V_query(P, V, tri, x0) / T_star(*x0))
    print('адаптив m=%d N/запрос≈%d: V/T* mean %.3f max %.3f' % (m, np.mean(Ns), np.mean(r), np.max(r)), flush=True)
