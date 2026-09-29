"""research: почему V по графу клеток (v6 value.solve_V) занижена и хуже при мельчении h; зависимость от τ.
Гипотеза: V = T* только Гёльдер-½ у кривой переключения (√ расстояния), оптимальные траектории финиша идут ВДОЛЬ неё →
каждый шаг Беллмана даёт ошибку интерполяции ~√h (вогнутость → занижение), шагов ~T/τ ⇒ суммарно ~√h·T/τ.
При τ = h/2 это ~h^(-1/2) — растёт при мельчении; лечение — τ ≫ √h (длинные клетки), либо цепочки спор вдоль траекторий."""
import sys, json, time, numpy as np
sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v6')
from src.atlas6.value import solve_V
from src.atlas6.agent import T_star
out = {}
for h in (0.4, 0.2, 0.1):
    for k, tau in (("h/2", h/2), ("h", h), ("2h", 2*h), ("sqrt h", np.sqrt(h))):
        t0 = time.time(); xs, V, it = solve_V(h, tau=tau)
        X, Vv = np.meshgrid(xs, xs, indexing='ij'); m = (np.abs(X) <= 2) & (np.abs(Vv) <= 2) & (V < 1e2)
        E = V[m] - np.vectorize(T_star)(X[m], Vv[m])
        out[f"h={h} tau={k}"] = dict(tau=float(tau), mean=float(E.mean()), min=float(E.min()), max=float(E.max()),
                                     cover=float(m.sum() / ((np.abs(X) <= 2) & (np.abs(Vv) <= 2)).sum()), iters=it, sec=round(time.time()-t0, 1))
        print(f"h={h} tau={k}", out[f"h={h} tau={k}"], flush=True)
json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)
