"""Атлас DI (source_doc §4–§6): перекрывающиеся клетки двух слоёв u=±umax. Спора — центр клетки; core = уменьшенная клетка (α·r, α·τ), halo — остальное.
Соседи — клетки с общими точками; условие покрытия — выход каждой клетки (центр торца и его концы) лежит в ядре какой-то клетки."""
import numpy as np
from .cell import Cell


def seed_lattice(xlim, vlim, dx, dv):
    xs = np.arange(xlim[0], xlim[1] + 1e-9, dx); vs = np.arange(vlim[0], vlim[1] + 1e-9, dv)
    return [(x, v) for x in xs for v in vs]


class Atlas:
    def __init__(self, centers, r=0.3, tau=0.5, alpha=0.7, Lx=1.0, Lv=1.0, umax=1.0):
        self.alpha = alpha
        self.cells = [Cell(c, u, r, tau, Lx, Lv) for c in centers for u in (+umax, -umax)]
        self.cores = [Cell(c.c, c.u, alpha * r, alpha * tau, Lx, Lv) for c in self.cells]

    def containing(self, q, core=False):
        """Индексы клеток, содержащих точку q (core=True — только ядра)."""
        cs = self.cores if core else self.cells
        return [i for i, c in enumerate(cs) if c.contains(q)]

    def sample(self, i, ns=5, nt=5):
        """Опорные точки клетки i (сетка (s, t) — для оценки перекрытий)."""
        c = self.cells[i]
        return [self._pt(c, s, t) for s in np.linspace(-c.r, c.r, ns) for t in np.linspace(-c.tau, c.tau, nt)]

    @staticmethod
    def _pt(c, s, t):
        from .cell import flow
        return flow(c.segment(np.array(s)), c.u, t)

    def neighbors(self):
        """Граф соседства: i—j, если хоть одна опорная точка i лежит в j (или наоборот)."""
        n = len(self.cells); nb = [set() for _ in range(n)]
        pts = [self.sample(i) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and self.cells[j].contains(pts[i][12]) or (i != j and any(self.cells[j].contains(p) for p in pts[i])):
                    nb[i].add(j); nb[j].add(i)
        return nb

    def exit_gaps(self):
        """Клетки, чей выход не лежит в ядре ни одной клетки (дыры — сюда сеять споры). Проверяются центр торца и его концы."""
        gaps = []
        for i, c in enumerate(self.cells):
            e = c.exit(); pts = [e[0], e[1], (e[0] + e[1]) / 2]
            if not all(self.containing(p, core=True) for p in pts):
                gaps.append(i)
        return gaps

    def coverage(self, xlim, vlim, n=40, core=False):
        """Доля точек прямоугольника, лежащих хотя бы в одной клетке (или ядре)."""
        xs = np.linspace(*xlim, n); vs = np.linspace(*vlim, n)
        return float(np.mean([bool(self.containing((x, v), core)) for x in xs for v in vs]))
