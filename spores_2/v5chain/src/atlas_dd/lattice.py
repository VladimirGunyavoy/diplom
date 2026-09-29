"""State lattice дифдрайва, схема A (A2): узлы (i, j, k) — позиция на решётке шага h и курс из HEADINGS.
Курсы — направления примитивных целочисленных векторов (|компонента| ≤ 2): прямая вдоль курса попадает точно в узел решётки.
Рёбра: поворот на месте между соседними курсами (cost = Δθ/ωmax), прямая ± к ближайшему узлу вдоль курса (cost = длина/vmax).
Дуги (одновременные v и ω) в A2 не входят — добавляются в A2b."""
import numpy as np
from math import gcd, atan2, hypot, pi
from src.atlas.solve import cost_to_go

VECS = sorted({(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0) and gcd(abs(a), abs(b)) == 1},
              key=lambda p: atan2(p[1], p[0]))
HEADINGS = [atan2(b, a) for a, b in VECS]          # 16 курсов, по возрастанию θ ∈ (−π, π]
NH = len(HEADINGS)


def heading_index(th):
    """Ближайший курс (по циклической разности) и остаток."""
    d = [abs((th - h + pi) % (2 * pi) - pi) for h in HEADINGS]
    k = int(np.argmin(d)); return k, d[k]


def build_edges(n, h=1.0, vmax=1.0, wmax=1.0):
    """Решётка [−n, n]² шага h. Возвращает (nodes: {key: (x, y, θ)}, edges: {key: [(key2, cost, mode)]}); mode: 'r+','r-','f','b'."""
    nodes, edges = {}, {}
    rng = range(-n, n + 1)
    for i in rng:
        for j in rng:
            for k in range(NH):
                nodes[(i, j, k)] = (i * h, j * h, HEADINGS[k])
    for (i, j, k) in nodes:
        e = []
        for dk, m in ((1, 'r+'), (-1, 'r-')):
            k2 = (k + dk) % NH
            dth = abs((HEADINGS[k2] - HEADINGS[k] + pi) % (2 * pi) - pi)
            e.append(((i, j, k2), dth / wmax, m))
        a, b = VECS[k]
        for sg, m in ((1, 'f'), (-1, 'b')):
            k2 = (i + sg * a, j + sg * b, k)
            if k2 in nodes:
                e.append((k2, hypot(a, b) * h / vmax, m))
        edges[(i, j, k)] = e
    return nodes, edges


def solve(n, h=1.0, vmax=1.0, wmax=1.0, goal_heading=0.0):
    nodes, edges = build_edges(n, h, vmax, wmax)
    kg, _ = heading_index(goal_heading)
    goal = (0, 0, kg)
    T, policy = cost_to_go(edges, goal)
    return nodes, edges, T, policy, goal


def extract_controls(edges, policy, start, goal):
    """A4 (узловой): последовательность управлений из узла start в goal по policy: [(mode, время)], mode: r+/r-/f/b.
    Подряд идущие одинаковые режимы сливаются. Запрос из произвольной точки (интерполяция) — отдельный шаг."""
    out, k = [], start
    while k != goal:
        m, k2 = policy[k]
        c = next(cost for kk, cost, mm in edges[k] if kk == k2 and mm == m)
        if out and out[-1][0] == m:
            out[-1] = (m, out[-1][1] + c)
        else:
            out.append((m, c))
        k = k2
    return out


def interp_T3(x, y, th, T, n, h=1.0):
    """Запрос T из произвольной точки: билинейно по (x, y) на двух соседних курсах, затем линейно по θ (циклически, курсы неравномерны).
    nan вне поля или при недостижимом угле."""
    fx, fy = x / h, y / h
    i, j = int(np.floor(fx)), int(np.floor(fy))
    if i < -n or j < -n or i >= n or j >= n:
        return np.nan
    u, w = fx - i, fy - j
    th = (th + pi) % (2 * pi) - pi
    k = int(np.searchsorted(HEADINGS, th, side='right')) - 1        # HEADINGS[k] ≤ th < HEADINGS[k+1]
    k1 = (k + 1) % NH
    hk = HEADINGS[k % NH]; hk1 = HEADINGS[k1] + (2 * pi if k1 == 0 or k < 0 else 0)
    if k < 0:                                                         # th левее первого курса: пара (последний, первый)
        k = NH - 1; hk = HEADINGS[k] - 2 * pi; hk1 = HEADINGS[0]; k1 = 0
    elif k == NH - 1:
        hk1 = HEADINGS[0] + 2 * pi
    s = (th - hk) / (hk1 - hk)
    def bil(kk):
        v = [T.get((i + a, j + b, kk)) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
        if any(t is None for t in v):
            return np.nan
        return (1 - u) * (1 - w) * v[0] + u * (1 - w) * v[1] + (1 - u) * w * v[2] + u * w * v[3]
    return (1 - s) * bil(k) + s * bil(k1)
