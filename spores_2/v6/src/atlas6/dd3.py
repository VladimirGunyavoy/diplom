"""Клетка v6 для дифдрайва (x, y, θ), слой = постоянные (v, ω): поток замкнутый (дуга), поле левоинвариантно на SE(2) (research/diffdrive_v6_plan.md).
Сегмент — параллелограмм в 2 направлениях ⟂ полю в метрике dx²+dy²+c²dθ²; e1 — боковое (левее курса) при v≠0, иначе ось x тела; e2 = F×e1. Стены — потоки 4 углов."""
import numpy as np


def flow(p, vw, t):
    """Поток позы p = (..., 3) при постоянных (v, ω) за t (замкнуто)."""
    p = np.asarray(p, float); v, w = vw; x, y, th = p[..., 0], p[..., 1], p[..., 2]
    if abs(w) < 1e-12:
        return np.stack([x + v * t * np.cos(th), y + v * t * np.sin(th), th + 0 * x], -1)
    return np.stack([x + v / w * (np.sin(th + w * t) - np.sin(th)), y - v / w * (np.cos(th + w * t) - np.cos(th)), th + w * t], -1)


def compose(q, p):
    """L_q p = q·p в SE(2): p = (x, y, θ) в системе q → в мировой."""
    q = np.asarray(q, float); p = np.asarray(p, float); c, s = np.cos(q[..., 2]), np.sin(q[..., 2])
    return np.stack([q[..., 0] + c * p[..., 0] - s * p[..., 1], q[..., 1] + s * p[..., 0] + c * p[..., 1], q[..., 2] + p[..., 2]], -1)


class Cell3:
    def __init__(self, center, vw, r, tau, c=1.0):
        self.q = np.asarray(center, float); self.vw, self.r, self.tau, self.c = vw, r, tau, c
        v, w = vw; th = self.q[2]
        Fn = np.array([v * np.cos(th), v * np.sin(th), c * w])
        lat = np.array([-np.sin(th), np.cos(th), 0.0]) if abs(v) > 1e-12 else np.array([np.cos(th), np.sin(th), 0.0])
        e2 = np.cross(Fn, lat); e2 /= np.linalg.norm(e2)
        self.e = [lat, e2]                                     # ортонормированные, ⟂ Fn (в координатах (x, y, cθ))

    def corners(self, s1=None, s2=None):
        r = self.r; out = []
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            d = a * r * self.e[0] + b * r * self.e[1]
            out.append(self.q + np.array([d[0], d[1], d[2] / self.c]))
        return np.array(out)

    def wall(self, k, t):
        return np.array([flow(self.corners()[k], self.vw, ti) for ti in np.atleast_1d(t)])

    def entry(self):
        return flow(self.corners(), self.vw, -self.tau)

    def exit(self):
        return flow(self.corners(), self.vw, self.tau)

    def stretch(self):
        """ρ = наибольшая длина ребра торца / длина ребра сегмента (метрика с c). Площадь не годится: поток сохраняет объём, сдвиг её не меняет."""
        m = np.array([1.0, 1.0, self.c]); L = lambda a, b: float(np.linalg.norm((a - b) * m))
        S, E = self.corners(), self.exit()
        return max(L(E[1], E[0]), L(E[3], E[0])) / L(S[1], S[0])
