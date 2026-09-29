"""Клетка SPORE v6 для двойного интегратора (ẋ = v, v̇ = u, u = ±umax): сегмент, стены, вход/выход, координаты клетки.
Поток при постоянном u аффинный: x(t) = x0 + v0 t + u t²/2, v(t) = v0 + u t — торцы прямые, стены параболы (source_doc §3).
Масштабы Lx, Lv — нормировка осей (§7): сегмент перпендикулярен полю в нормированных координатах, r — полуширина в них же."""
import numpy as np


def flow(p, u, t):
    """Точка(и) p = (x, v) → через время t (t<0 — назад). p: (..., 2), t: скаляр или (...)."""
    p = np.asarray(p, float); t = np.asarray(t, float)
    return np.stack([p[..., 0] + p[..., 1] * t + u * t ** 2 / 2, p[..., 1] + u * t], axis=-1)


class Cell:
    def __init__(self, center, u, r, tau, Lx=1.0, Lv=1.0):
        self.c = np.asarray(center, float); self.u, self.r, self.tau = float(u), float(r), float(tau)
        self.Lx, self.Lv = float(Lx), float(Lv)
        Fn = np.array([self.c[1] / Lx, u / Lv])                      # поле в нормированных координатах
        nn = np.array([-Fn[1], Fn[0]]) / np.hypot(*Fn)               # единичный перпендикуляр там
        self.d = np.array([nn[0] * Lx, nn[1] * Lv])                  # он же в физических координатах (направление сегмента)

    def segment(self, s):
        """Точки сегмента, s ∈ [−r, r] (в нормированной длине вдоль перпендикуляра)."""
        return self.c + np.asarray(s, float)[..., None] * self.d

    def wall(self, sign, t):
        """Стена: траектория края сегмента sign·r (sign=±1), t ∈ [−τ, τ]."""
        return flow(self.segment(sign * self.r), self.u, t)

    def entry(self):
        return flow(self.segment(np.array([-self.r, self.r])), self.u, -self.tau)

    def exit(self):
        return flow(self.segment(np.array([-self.r, self.r])), self.u, self.tau)

    def _norm_len(self, ends):
        e = ends[1] - ends[0]
        return float(np.hypot(e[0] / self.Lx, e[1] / self.Lv))

    def stretch(self, t=None, physical=False):
        """ρ = длина торца после потока t / длина сегмента (по умолчанию t = τ). physical=True — метрика без нормировки (как в §3 документа)."""
        t = self.tau if t is None else t
        e = flow(self.segment(np.array([-self.r, self.r])), self.u, t); s0 = self.segment(np.array([-self.r, self.r]))
        if physical:
            return float(np.linalg.norm(e[1] - e[0]) / np.linalg.norm(s0[1] - s0[0]))
        return self._norm_len(e) / self._norm_len(s0)

    def coords(self, q):
        """Координаты клетки точки q = (x, v): (s, t) — поперечное положение и время от сегмента вдоль потока (q = flow(segment(s), t)).
        t — корень квадратного уравнения cross(flow(q, −t) − c, d) = 0 с наименьшим |t|. Нет корня — None."""
        q = np.asarray(q, float); u = self.u; d = self.d; c = self.c
        # flow(q, −t) = (x − v t + u t²/2, v − u t); cross(w, d) = w0 d1 − w1 d0
        a = u * d[1] / 2
        b = -q[1] * d[1] + u * d[0]
        cc = (q[0] - c[0]) * d[1] - (q[1] - c[1]) * d[0]
        if abs(a) > 1e-14:
            D = b * b - 4 * a * cc
            if D < 0:
                return None
            sq = D ** 0.5; roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
        elif abs(b) > 1e-14:
            roots = [-cc / b]
        else:
            roots = []
        if not roots:
            return None
        t = min(roots, key=abs); w = flow(q, u, -t)
        s = float(np.dot(w - c, d) / np.dot(d, d))
        return s, t

    def contains(self, q):
        st = self.coords(q)
        return st is not None and abs(st[0]) <= self.r and -self.tau <= st[1] <= self.tau
