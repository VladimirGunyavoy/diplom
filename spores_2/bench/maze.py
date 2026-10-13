"""CA-MAZE: лабиринт (совершенный, DFS, сетка NxN, сид фиксирован) как SDF-картинка + дифдрайв 3D (ẋ=v cosθ, ẏ=v sinθ, θ̇=ω, |v|+|ω|≤1).
Клетка 1×1, стена толщиной WT; свободная ширина коридора 1−WT. Цель — центр клетки goal (смещаем мир так, чтобы цель была в нуле, как у dd). gfree(y) = SDF(x,y) − RROB (>0 свободно).
Эталон T_ref = «лучшее известное»: геодезический путь по сетке (A* 8-связный по SDF-карте) → натяжение (visibility shortcut) → T = Σдлин + Σ|поворотов| (rotate–straight–rotate, допустимо для дд)."""
import os, heapq, numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
import common
PI = np.pi
N_CELLS = 6; WT = .2; RROB = .1; PXM = .02

def make_maze(n=N_CELLS, seed=0):
    rng = np.random.default_rng(seed); vis = np.zeros((n, n), bool); walls = {}   # walls[(i,j,d)] открыт; d: 0 вправо(+x),1 вверх(+y)
    openE = set(); st = [(0, 0)]; vis[0, 0] = True
    while st:
        i, j = st[-1]; nb = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + di < n and 0 <= j + dj < n and not vis[i + di, j + dj]]
        if not nb: st.pop(); continue
        a = nb[rng.integers(len(nb))]; vis[a] = True; openE.add(frozenset(((i, j), a))); st.append(a)
    return openE

def build(n=N_CELLS, seed=0, wt=WT):
    openE = make_maze(n, seed); S = int(round(n / PXM)); wall = np.zeros((S, S), bool); h = wt / 2
    def rect(x0, x1, y0, y1, v=True): wall[max(0, int(round(y0 / PXM))):int(round(y1 / PXM)), max(0, int(round(x0 / PXM))):int(round(x1 / PXM))] = v
    for k in range(n + 1): rect(k - h, k + h, 0, n); rect(0, n, k - h, k + h)
    for e in openE:
        (i, j), (i2, j2) = sorted(tuple(e))
        if i2 != i: rect(i2 - h, i2 + h, j + h, j + 1 - h, False)
        else: rect(i + h, i + 1 - h, j2 - h, j2 + h, False)
    free = ~wall; sdf = distance_transform_edt(free) * PXM - distance_transform_edt(wall) * PXM      # >0 свободно, расстояние до стены (м), внутри стен <0
    return openE, wall, sdf

class Maze:
    def __init__(s, n=N_CELLS, seed=0, goal=(5, 5)):
        s.n = n; s.openE, s.wall, s.sdf = build(n, seed); s.G0 = np.array([goal[0] + .5, goal[1] + .5])      # центр клетки цели (мир до сдвига)
    def gfree(s, y):
        y = np.asarray(y, float); px = (y[..., 0] + s.G0[0]) / PXM - .5; py = (y[..., 1] + s.G0[1]) / PXM - .5
        return map_coordinates(s.sdf, [py.ravel(), px.ravel()], order=1, mode='constant', cval=-5.).reshape(px.shape) - RROB

class MazeDD(common.DD):
    """дд в лабиринте, цель в нуле (центр клетки goal), коробка цели ±.05; выход за коробку = столкновение со стеной (gfree<0)"""
    name = 'maze'
    def __init__(s, nst=50, n=N_CELLS, seed=0):
        s.M = Maze(n, seed); s.lo = np.array([-n, -n, -PI]); s.hi = np.array([n, n, PI]); s.fl_lo = s.lo.copy(); s.fl_hi = s.hi.copy()
        rng = np.random.default_rng(1); X = []
        while len(X) < nst:
            p = rng.uniform(0, n, 2) - s.M.G0; 
            if s.M.gfree(p[None])[0] > .05 and np.linalg.norm(p) > 1.5: X.append([p[0], p[1], rng.uniform(-PI, PI)])
        s.X0 = np.array(X); s.keep = np.ones(nst, bool); s.Tref = np.array([ref_T(s, x) for x in s.X0]); s.keep = np.isfinite(s.Tref)
    def gfree(s, Y): return s.M.gfree(np.asarray(Y)[..., :2])
    def out_of_box(s, Y): return s.gfree(Y) < 0

def _astar(free, s, g):
    S = free.shape[0]; d = {s: 0.}; pq = [(0., s)]; par = {}
    nb = [(1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.), (1, 1, 2 ** .5), (1, -1, 2 ** .5), (-1, 1, 2 ** .5), (-1, -1, 2 ** .5)]
    while pq:
        c, u = heapq.heappop(pq)
        if u == g: break
        if c > d[u] + np.hypot(u[0] - g[0], u[1] - g[1]) + 1e-9: continue
        for di, dj, w in nb:
            v = (u[0] + di, u[1] + dj)
            if 0 <= v[0] < S and 0 <= v[1] < S and free[v] and d[u] + w < d.get(v, 1e18):
                d[v] = d[u] + w; par[v] = u; heapq.heappush(pq, (d[v] + np.hypot(v[0] - g[0], v[1] - g[1]), v))
    path = [g]
    while path[-1] != s: path.append(par[path[-1]])
    return path[::-1]

def ref_T(env, y0, margin=.02):
    """T_ref (верхняя оценка «лучшего известного»): A* по пикселям → натяжение (прямая видимость, gfree ≥ margin) → T = Σ длин + Σ|поворотов| (начальный курс y0[2], конечный 0)"""
    M = env.M; free = M.sdf > RROB + margin; ij = lambda p: (int(round((p[1] + M.G0[1]) / PXM - .5)), int(round((p[0] + M.G0[0]) / PXM - .5)))
    s = ij(y0[:2]); g = ij(np.zeros(2))
    if not free[s]: return np.nan
    P = [np.array([(c[1] + .5) * PXM - M.G0[0], (c[0] + .5) * PXM - M.G0[1]]) for c in _astar(free, s, g)]
    def los(a, b):
        t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / .01)))[:, None]; return (M.gfree(a + t * (b - a)) >= margin - RROB + 1e-9).all()
    Q = [P[0]]; i = 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not los(P[i], P[j]): j -= 1
        Q.append(P[j]); i = j
    Q = np.array(Q); seg = np.diff(Q, axis=0); L = np.linalg.norm(seg, axis=1).sum(); hd = np.arctan2(seg[:, 1], seg[:, 0]); h = np.r_[y0[2], hd, 0.]
    turn = np.abs(np.angle(np.exp(1j * np.diff(h)))).sum(); return float(L + turn)
