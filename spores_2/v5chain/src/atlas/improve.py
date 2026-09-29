"""Улучшения АТЛАС §7: складка v=0, излом, адаптивная сетка, vmax, цель-множество."""
import numpy as np
from collections import defaultdict
from .coords import A, c_plus, c_minus
from .lattice import key, build_lattice, build_edges
from .solve import cost_to_go
from .interp import interp_T


def make_grid(lo, hi, n, extra=()):
    g = np.unique(np.round(np.concatenate([np.linspace(lo, hi, n), np.asarray(extra, float)]), 12))
    return g


class Atlas:
    def __init__(self, C, D, a=A, goals=None, vmax=None):
        self.C, self.D, self.a, self.vmax = C, D, a, vmax
        self.nodes = build_lattice(C, D, a, vmax)
        self.edges = build_edges(self.nodes, a)
        if vmax is not None:
            self._add_vmax_lines()
        if goals is None:
            i0 = int(np.argmin(abs(C))); goals = [key(i0, int(np.argmin(abs(D))), 0, C, D)]
        self.goals = goals
        self.T, self.policy = cost_to_go(self.edges, goals if len(goals) > 1 else goals[0])
        self._par = None

    def _add_vmax_lines(self):
        """Режим a=0 на v=±vmax: узлы (i,None,±2) от c-парабол, (None,j,±2) от d-парабол (АТЛАС §7)."""
        a, vm, nodes, edges = self.a, self.vmax, self.nodes, self.edges
        for sg in (+1, -1):
            v = sg * vm
            line = []
            for i, c in enumerate(self.C):
                nodes[(i, None, 2 * sg)] = (c + v * v / (2 * a), v); line.append((nodes[(i, None, 2 * sg)][0], (i, None, 2 * sg)))
            for j, d in enumerate(self.D):
                nodes[(None, j, 2 * sg)] = (d - v * v / (2 * a), v); line.append((nodes[(None, j, 2 * sg)][0], (None, j, 2 * sg)))
            line.sort(key=lambda t: t[0], reverse=(sg < 0))   # вдоль дрейфа
            for (x1, k1), (x2, k2) in zip(line, line[1:]):
                edges[k1].append((k2, abs(x2 - x1) / vm, 0))
                if abs(x2 - x1) < 1e-9:            # совпавшие узлы c- и d-парабол — переход в обе стороны
                    edges[k2].append((k1, 0.0, 0))
        # параболы c_i/d_j доходят до линий: добавить в рёбра вдоль парабол
        plus, minus = defaultdict(list), defaultdict(list)
        for k, (x, v) in nodes.items():
            i, j, s = k
            if abs(s) == 2:
                (plus[i] if j is None else minus[j]).append((v, k))
            else:
                plus[i].append((v, k)); minus[j].append((v, k))
        for k in list(edges):                      # пересобрать рёбра парабол с учётом линий
            edges[k] = [e for e in edges[k] if e[2] == 0]
        for lst in plus.values():
            lst.sort(key=lambda t: t[0])
            for (v1, k1), (v2, k2) in zip(lst, lst[1:]):
                edges[k1].append((k2, (v2 - v1) / a, +1))
        for lst in minus.values():
            lst.sort(key=lambda t: t[0], reverse=True)
            for (v1, k1), (v2, k2) in zip(lst, lst[1:]):
                edges[k1].append((k2, (v1 - v2) / a, -1))

    # --- запросы ---
    def bilinear(self, x, v):
        return interp_T(x, v, self.C, self.D, self.T, self.a)

    def _parabolas(self):
        if self._par is None:
            par = defaultdict(list)
            for k, t in self.T.items():
                if k[0] is not None and abs(k[2]) != 2:
                    par[k[0]].append((self.nodes[k][1], t))
            self._par = {i: tuple(np.array(z) for z in zip(*sorted(l))) for i, l in par.items()}
        return self._par

    def cv(self, x, v):
        """Интерполяция в (c, v): вдоль парабол c_i линейно по v, затем по c — гладко через ось v=0."""
        c = c_plus(x, v, self.a); C = self.C
        i = np.searchsorted(C, c) - 1
        if i < 0 or i + 1 >= len(C):
            return np.nan
        par = self._parabolas()
        if i not in par or i + 1 not in par:
            return np.nan
        t0 = np.interp(v, *par[i], left=np.nan, right=np.nan)
        t1 = np.interp(v, *par[i + 1], left=np.nan, right=np.nan)
        u = (c - C[i]) / (C[i + 1] - C[i])
        return (1 - u) * t0 + u * t1

    def hybrid(self, x, v, thr=None):
        """thr=None: (c,v) только там, где билинейная не определена; thr=числу: (c,v) при |v|<thr."""
        if thr is not None and abs(v) < thr:
            t = self.cv(x, v)
            if not np.isnan(t):
                return t
        t = self.bilinear(x, v)
        return self.cv(x, v) if np.isnan(t) else t

    def bellman(self, x, v, base=None):
        """Излом: T(q) = min по режимам ±a (шаг до ближайшей параболы сетки другого семейства + интерполянт там)."""
        base = base or self.hybrid
        a, C, D, eps = self.a, self.C, self.D, 1e-9
        c, d = c_plus(x, v, a), c_minus(x, v, a)
        best = np.inf
        # режим +a: c=const, v растёт; d(v)=c+a v²... в единицах a: d-c = v²/a
        if v >= 0:
            cand = D[D > d + eps]; v2 = np.sqrt(a * (cand[0] - c)) if len(cand) else None
        else:
            cand = D[(D < d - eps) & (D >= c)]; v2 = -np.sqrt(a * (cand[-1] - c)) if len(cand) else 0.0
        if v2 is not None and v2 > v + eps:
            t = base(c + v2 * v2 / (2 * a), v2)
            if not np.isnan(t):
                best = min(best, (v2 - v) / a + t)
        # режим −a: d=const, v убывает
        if v > 0:
            cand = C[(C > c + eps) & (C <= d)]; v2 = np.sqrt(a * (d - cand[0])) if len(cand) else 0.0
        else:
            cand = C[C < c - eps]; v2 = -np.sqrt(a * (d - cand[-1])) if len(cand) else None
        if v2 is not None and v2 < v - eps:
            t = base(d - v2 * v2 / (2 * a), v2)
            if not np.isnan(t):
                best = min(best, (v - v2) / a + t)
        return best if np.isfinite(best) else base(x, v)


def sample_points(n=2000, seed=0, xr=1.5, vr=1.5):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-xr, xr), rng.uniform(-vr, vr)) for _ in range(n)]


def metrics(f, ref, pts):
    e_all, e_near, e_far = [], [], []
    for x, v in pts:
        t = f(x, v)
        if np.isnan(t):
            continue
        e = abs(t - ref(x, v)); e_all.append(e); (e_near if abs(v) < 0.3 else e_far).append(e)
    m = lambda z: float(np.mean(z)) if z else None
    mx = lambda z: float(np.max(z)) if z else None
    return dict(n=len(e_all), mean=m(e_all), max=mx(e_all), mean_near=m(e_near), mean_far=m(e_far), max_near=mx(e_near), max_far=mx(e_far))


def refine(C, D, ref, tol, lo=-1.5, hi=1.5, a=A):
    """Дробление ячеек по ошибке T в центре: вставляет середины C и D ячейки (тензорно). Возвращает новые C, D и число дроблений."""
    at = Atlas(C, D, a)
    addC, addD = [], []
    for i in range(len(C) - 1):
        for j in range(len(D) - 1):
            cc, dc = (C[i] + C[i + 1]) / 2, (D[j] + D[j + 1]) / 2
            if dc > cc:
                pts = [((cc + dc) / 2, s * np.sqrt(a * (dc - cc))) for s in (1, -1)]
            else:
                pts = [((cc + dc) / 2, 0.0)]
            for x, v in pts:
                if abs(x) > hi or abs(v) > hi:
                    continue
                t = at.hybrid(x, v)
                if not np.isnan(t) and abs(t - ref(x, v)) > tol:
                    addC.append(cc); addD.append(dc); break
    return (np.unique(np.round(np.concatenate([C, addC]), 12)),
            np.unique(np.round(np.concatenate([D, addD]), 12)), len(addC))
