"""CA-MAZE 4D: лабиринт maze.py + дд с инерцией по скорости (x, y, θ, v): ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω, v̇ = a − v; управление (a, ω), |a| + |ω| ≤ 1 (ромб, как у дд), |v| ≤ 1.
Старты те же, что у 3D (rng(1)), v0 = 0. Цель — квадрат ±GB по (x, y), θ и v любые (как у Car). T_ref здесь = НИЖНЯЯ граница T_lb = длина кратчайшего пути (A* + натяжение) / v_max = L (|v| ≤ 1):
не достижима (разгон, повороты), поэтому в таблице — T/T_lb, плюс лучшее из методов. Блок SYS=maze4 для growN: mk_maze4_src.py."""
import numpy as np, common, maze
PI = np.pi; GB = .08

def path_len(env, y0, margin=.02):
    """длина кратчайшего пути в лабиринте: A* по пикселям → натяжение (прямая видимость), без поворотов"""
    M = env.M; free = M.sdf > maze.RROB + margin; ij = lambda p: (int(round((p[1] + M.G0[1]) / maze.PXM - .5)), int(round((p[0] + M.G0[0]) / maze.PXM - .5)))
    s = ij(y0[:2]); g = ij(np.zeros(2))
    if not free[s]: return np.nan
    P = [np.array([(c[1] + .5) * maze.PXM - M.G0[0], (c[0] + .5) * maze.PXM - M.G0[1]]) for c in maze._astar(free, s, g)]
    def los(a, b):
        t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / .01)))[:, None]; return (M.gfree(a + t * (b - a)) >= margin - 1e-9).all()
    Q = [P[0]]; i = 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not los(P[i], P[j]): j -= 1
        Q.append(P[j]); i = j
    return float(np.linalg.norm(np.diff(np.array(Q), axis=0), axis=1).sum())

class MazeCar(common.Env):
    name = 'maze4'; n = 4; m = 2; period = np.array([0., 0., 2 * PI, 0.]); umin = np.array([-1., -1.]); umax = np.array([1., 1.]); half = np.array([GB, GB, PI, 1.])      # half — для сэмплера цели OMPL
    def __init__(s, nst=50, n=maze.N_CELLS, seed=0):
        s.M = maze.Maze(n, seed); s.lo = np.array([-n, -n, -PI, -1.]); s.hi = np.array([n, n, PI, 1.]); s.fl_lo = s.lo.copy(); s.fl_hi = s.hi.copy()
        rng = np.random.default_rng(1); X = []
        while len(X) < nst:
            p = rng.uniform(0, n, 2) - s.M.G0
            if s.M.gfree(p[None])[0] > .05 and np.linalg.norm(p) > 1.5: X.append([p[0], p[1], rng.uniform(-PI, PI), 0.])
        s.X0 = np.array(X); s.Tref = np.array([path_len(s, x) for x in s.X0]); s.keep = np.isfinite(s.Tref)
    def f(s, Y, u): s.nf += 1; Y = np.asarray(Y, float); u = np.asarray(u, float); th, v = Y[..., 2], Y[..., 3]; return np.stack([v * np.cos(th), v * np.sin(th), u[..., 1] + 0 * th, u[..., 0] - v], -1)
    def clip_u(s, u):
        u = np.clip(u, -1, 1); a = np.abs(u).sum(-1, keepdims=True); return np.where(a > 1, u / np.maximum(a, 1e-12), u)
    def gfree(s, Y): return s.M.gfree(np.asarray(Y)[..., :2])
    def in_goal(s, Y): Y = np.asarray(Y); return (np.abs(Y[..., 0]) <= GB + 1e-6) & (np.abs(Y[..., 1]) <= GB + 1e-6)
    def goal_dist(s, Y): Y = np.asarray(Y); return np.maximum(np.abs(Y[..., 0]), np.abs(Y[..., 1])) / GB - 1.
    def out_of_box(s, Y): Y = np.asarray(Y); return (s.gfree(Y) < 0) | (np.abs(Y[..., 3]) > 1 + 1e-6)
