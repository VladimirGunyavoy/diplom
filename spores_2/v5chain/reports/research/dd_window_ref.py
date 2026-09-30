"""research: эталон дифдрайва (ромб-U) «до ОКНА цели» (|xy| ≤ R, |θ| ≤ Rθ), а не до точки — для честной сверки атласа с окном цели.
ref_window(p) = min по позам g окна (сетка: nr колец × na углов по xy, nt по θ) времени TGT из p в g (перевод p в систему g).
Плюс TGTGT в центр окна (змейка в точку иногда лучше TGT в окно: без неё окно/точка до 1.087).
Верхняя оценка истинного минимума (дискретность выборки окна); с ростом nr/na/nt → сверху."""
import numpy as np, importlib.util
sp = importlib.util.spec_from_file_location('ref', __file__.replace('dd_window_ref.py', 'dd_rhombus_ref.py')); ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
def to_frame(p, g):                                   # поза p в системе позы g
    c, s = np.cos(g[2]), np.sin(g[2]); dx, dy = p[0]-g[0], p[1]-g[1]
    return (c*dx + s*dy, -s*dx + c*dy, ref.wrap(p[2]-g[2]))
def ref_window(p, R=0.25, Rth=0.26, nr=4, na=12, nt=5, point=True):
    G = [(0.0, 0.0, t) for t in np.linspace(-Rth, Rth, nt)]
    G += [(r*np.cos(a), r*np.sin(a), t) for r in np.linspace(R/nr, R, nr) for a in np.linspace(0, 2*np.pi, na, endpoint=False) for t in np.linspace(-Rth, Rth, nt)]
    w = min(ref.tgt(*to_frame(p, g)) for g in G)
    return min(w, ref.tgtgt(*p)) if point else w      # центр окна — тоже поза окна: змейка в точку бывает лучше TGT в окно
if __name__ == "__main__":
    rng = np.random.default_rng(3); P = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(20)]
    pt = np.array([min(ref.tgt(*p), ref.tgtgt(*p)) for p in P]); w = np.array([ref_window(p) for p in P]); w2 = np.array([ref_window(p, nr=8, na=24, nt=9, point=False) for p in P]); w1 = np.array([ref_window(p, point=False) for p in P])
    print(f"окно/точка: mean {np.mean(w/pt):.3f} min {np.min(w/pt):.3f} max {np.max(w/pt):.3f}; выборка ×8 плотнее меняет на max {np.max(np.abs(w2-w1)):.3f}")


def tgtgt_window(p, R=0.25, Rth=0.26, starts=40, rng=None):
    """Змейка TGTGT прямо в ОКНО (неравенства |xy_конец| ≤ R, |sin θ_конец| ≤ sin Rθ, cos > 0), SLSQP, многостарт — точнее центра окна."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(0) if rng is None else rng
    def end(z):
        q = tuple(p)
        for k, val in zip('TGTGT', z): q = ref.pose(q, (k, val))
        return q
    cost = lambda z: np.sum(np.sqrt(z**2 + 1e-9))
    cons = [{'type': 'ineq', 'fun': lambda z: R**2 - end(z)[0]**2 - end(z)[1]**2},
            {'type': 'ineq', 'fun': lambda z: np.sin(Rth)**2 - np.sin(end(z)[2])**2},
            {'type': 'ineq', 'fun': lambda z: np.cos(end(z)[2])}]
    best = np.inf
    for _ in range(starts):
        r = minimize(cost, rng.uniform(-3, 3, 5), constraints=cons, method='SLSQP', options=dict(maxiter=300))
        e = end(r.x)
        if np.hypot(e[0], e[1]) <= R + 1e-6 and abs(ref.wrap(e[2])) <= Rth + 1e-6: best = min(best, cost(r.x))
    return best


def ref_window2(p, R=0.25, Rth=0.26):
    return min(ref_window(p, R, Rth, point=False), tgtgt_window(p, R, Rth))


if __name__ == "__main__":
    P2 = P[:10]; a = np.array([ref_window(p) for p in P2]); b = np.array([ref_window2(p) for p in P2])
    print(f"ref_window2 / ref_window (змейка в окно vs в центр): mean {np.mean(b/a):.3f} min {np.min(b/a):.3f}")
