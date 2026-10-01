"""PLAN 0з п.5/5а: эллиптический спавн + общий пул. Запрос (s, g) в норм. координатах; эллипс |p−s|+|p−g| ≤ L (a+b=const, все точки равноправны),
L = f·|s−g| растёт по шагам; на каждом шаге споры сеются РАВНОМЕРНО по площади/объёму КОРОЧКИ (между прежним и новым эллипсом, не на границе, п.5а),
в каждой точке — ПРЯМАЯ (поток вперёд) и ОБРАТНАЯ (поток назад) клетка в каждом слое, если точка не покрыта ядром клеток пула (k, dir). Пул общий между запросами."""
import numpy as np
from .celln import SysN, CellN


def reverse(S):
    return SysN(S.name + '_rev', S.n, S.K, lambda y, k, t: S.flow(y, k, -t), S.box, per=S.per)


def ring_points(S, rng, s, g, L0, L1, N):
    """N равномерных точек коробки в корочке L0 < |p−s|+|p−g| ≤ L1 (отбор из коробки; периодичность — через wrap расстояний)."""
    out = []; lo, hi = S.box
    while len(out) < N:
        P = lo + (hi - lo) * rng.random((20 * N, S.n))
        L = np.linalg.norm(S.wrap(P - s), axis=1) + np.linalg.norm(S.wrap(P - g), axis=1)
        out += list(P[(L > L0) & (L <= L1)])
        if len(out) == 0 and L1 > 20: break
    return np.array(out[:N])


class Pool:
    def __init__(self, S, seed=0):
        self.S, self.R = S, reverse(S); self.cells = {(k, d): [] for k in range(S.K) for d in (0, 1)}; self.rng = np.random.default_rng(seed)
    def covered(self, P, k, d):
        cov = np.zeros(len(P), bool)
        for C in self.cells[(k, d)]: cov[~cov] |= C.locate(P[~cov], 1.0)[2] if (~cov).any() else False
        return cov
    def size(self): return sum(len(v) for v in self.cells.values())
    def query(self, s, g, factors=(1.05, 1.2, 1.5, 2.0), N=40):
        """Растить эллипс; вернуть статистику: новых клеток по шагам, доля уже покрытых споровых точек (повторное использование пула)."""
        S = self.S; d = float(np.linalg.norm(S.wrap(s - g))); L0 = d; new = []; reuse = []; before = self.size()
        for f in factors:
            L1 = f * d; P = ring_points(S, self.rng, s, g, L0, L1, N); added = 0; hit = 0; tot = 0
            for k in range(S.K):
                for dr, SS in ((0, S), (1, self.R)):
                    cov = self.covered(P, k, dr); hit += cov.sum(); tot += len(P)
                    for p in P[~cov]:
                        if self.covered(p[None], k, dr)[0]: continue          # покрыта клеткой, созданной в этом же шаге
                        self.cells[(k, dr)].append(CellN(SS, k, p, rng=self.rng)); added += 1
            new.append(added); reuse.append(hit / tot); L0 = L1
        return dict(new=new, reuse=reuse, pool=self.size(), grew=self.size() - before)
