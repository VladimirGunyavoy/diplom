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


def arc_sweeps(nodes, T, n, h=1.0, vmax=1.0, wmax=1.0, iters=50, tol=1e-9, goal=None):
    """A2b: дуги (одновременные v, ω) между соседними курсами. Конец дуги — точно (Arc.from_chart), но не в узле →
    T в конце берётся через interp_T3 на курсе назначения; итерации Беллмана (T только уменьшается) от решения Дейкстра.
    Возвращает (новый T, число итераций, [изменение за итерацию])."""
    from .modes import Arc
    arcs = {}
    for key_, (x, y, th) in nodes.items():
        k = key_[2]; lst = []
        for dk in (1, -1):
            k2 = (k + dk) % NH
            dth = (HEADINGS[k2] - HEADINGS[k] + pi) % (2 * pi) - pi          # знак = направление вращения
            for sv in (1, -1):
                m = Arc(sv, 1 if dth > 0 else -1, vmax, wmax)
                lab, _ = m.to_chart((x, y, th))
                xe, ye, _ = m.from_chart(lab, th + dth)
                lst.append((abs(dth) / wmax, xe, ye, k2))
        arcs[key_] = lst
    T = dict(T); hist = []
    for it in range(iters):
        ch = 0.0
        for key_, lst in arcs.items():
            if key_ == goal:
                continue
            best = T[key_]
            for c, xe, ye, k2 in lst:
                t = interp_T3(xe, ye, HEADINGS[k2], T, n, h)
                if not np.isnan(t) and c + t < best - 1e-12:
                    best = c + t
            if best < T[key_]:
                ch = max(ch, T[key_] - best); T[key_] = best
        hist.append(ch)
        if ch < tol:
            break
    return T, len(hist), hist


def arc_edges(nodes, edges, h=1.0, vmax=1.0, wmax=1.0, M=4):
    """A2c: дуги с концом ТОЧНО в узле (без интерполяции). Дуга между курсами k→k2 (поворот Δ<π, ω=±ωmax, время Δ/ωmax): хорда идёт вдоль
    среднего курса φ и равна L = 2R·sin(Δ/2), R = |v|/ωmax; |v| ≤ vmax можно занижать (время то же) → хорда любой длины ≤ 2 sin(Δ/2)·vmax/ωmax.
    Берём пары (k,k2), где φ (или φ+π для v<0) совпадает с направлением целочисленного вектора w=(p,q), |p|,|q|≤M, и длины m·h·|w|, m·w допустимой.
    Возвращает новую копию edges с рёбрами mode 'a'."""
    W = [(p, q) for p in range(-M, M + 1) for q in range(-M, M + 1) if (p, q) != (0, 0) and gcd(abs(p), abs(q)) == 1]
    dirs = [(atan2(q, p), p, q) for p, q in W]
    new = {k: list(v) for k, v in edges.items()}
    pairs = []
    for k in range(NH):
        for k2 in range(NH):
            if k2 == k:
                continue
            dth = (HEADINGS[k2] - HEADINGS[k] + pi) % (2 * pi) - pi             # знак = направление вращения
            if abs(abs(dth) - pi) < 1e-9:
                continue
            phi = HEADINGS[k] + dth / 2
            for sv in (1, -1):
                ang = phi + (pi if sv < 0 else 0)
                for a, p, q in dirs:
                    if abs((a - ang + pi) % (2 * pi) - pi) < 1e-9:
                        Lmax = 2 * np.sin(abs(dth) / 2) * vmax / wmax
                        for m in range(1, 2 * M + 1):
                            L = m * h * hypot(p, q)
                            if L <= Lmax + 1e-12:
                                pairs.append((k, k2, m * p, m * q, abs(dth) / wmax))
    seen = set()
    for k, k2, dx, dy, c in pairs:
        if (k, k2, dx, dy) in seen:
            continue
        seen.add((k, k2, dx, dy))
        for (i, j, kk) in nodes:
            if kk != k:
                continue
            # хорда в сетке шагов h: dx, dy — в единицах h
            t = (i + dx, j + dy, k2)
            if t in nodes:
                new[(i, j, k)].append((t, c, 'a'))
    return new, len(seen)
