"""Агент v6 (DI): V хранится по клеткам в 2n+1 = 5 точках (спора, ±r поперёк, ±τ по потоку); ∇V — МНК по ним; u = −sign(∂V/∂v) (принцип максимума);
внутри клетки цели радиуса R_goal — точное правило (кривая x = −v|v|/2). Проверка: research/di_gradV_switch.md (без клетки цели зависает)."""
import numpy as np
from .cell import Cell, flow


def T_star(x, v, umax=1.0):
    """Точное время быстродействия DI в начало координат (эталон и V первой версии; V из атласа — следующий шаг).
    s = x + v|v|/2umax — положение относительно кривой переключения: s>0 → сначала u=−umax, s<0 → u=+umax, затем обратное."""
    s = x + v * abs(v) / (2 * umax)
    if abs(s) < 1e-12:
        return abs(v) / umax
    sg = 1.0 if s > 0 else -1.0
    return (sg * v + 2 * np.sqrt(max(v * v / 2 + sg * x * umax, 0.0) * umax)) / umax


class CellV:
    """Клетка с запомненными V в 5 точках и МНК-градиентом."""
    def __init__(self, cell, V=T_star):
        self.cell = cell; c = cell
        self.pts = np.array([c.c, c.segment(-c.r), c.segment(c.r), flow(c.c, c.u, -c.tau), flow(c.c, c.u, c.tau)])
        self.vals = np.array([V(p[0], p[1]) for p in self.pts])
        A = np.c_[np.ones(5), self.pts - c.c]
        self.grad = np.linalg.lstsq(A, self.vals, rcond=None)[0][1:]      # (∂V/∂x, ∂V/∂v)


def make_cells(h, xlim=(-4.0, 4.0), vlim=(-4.0, 4.0), r=None, tau=None, layer_u=+1.0, V=T_star):
    """Споры на решётке шага h; r = τ = h/2 по умолчанию (как в research)."""
    r = h / 2 if r is None else r; tau = h / 2 if tau is None else tau
    xs = np.arange(xlim[0], xlim[1] + 1e-9, h); vs = np.arange(vlim[0], vlim[1] + 1e-9, h)
    return [CellV(Cell((x, v), layer_u, r, tau), V) for x in xs for v in vs]


def run_agent(x, v, cells, R_goal, dt=0.01, tol=0.05, umax=1.0):
    """Агент из (x, v): (время/T*, дошёл, число переключений)."""
    C = np.array([cv.cell.c for cv in cells]); G = np.array([cv.grad for cv in cells])
    T0 = T_star(x, v); t = 0.0; sw = 0; uprev = 0.0
    while t < 3 * T0 + 1:
        if np.hypot(x, v) < tol:
            return t / T0, True, sw
        if np.hypot(x, v) < R_goal:
            s = x + v * abs(v) / 2; u = -np.sign(s) if abs(s) > 1e-9 else -np.sign(v)
        else:
            i = int(np.argmin((C[:, 0] - x) ** 2 + (C[:, 1] - v) ** 2)); u = -np.sign(G[i, 1]) or 1.0
        if u != uprev:
            sw += 1; uprev = u
        x, v = flow(np.array([x, v]), u * umax, dt); t += dt
    return t / T0, False, sw
