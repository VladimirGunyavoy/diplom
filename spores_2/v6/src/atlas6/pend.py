"""Маятник v6 (θ̈ = −sin θ + u, |u| ≤ umax < 1 — раскачка): V по сетке спор (θ периодично), цель (π, 0) — клетка радиуса R_goal.
Q_k(c) = τ + V(выход клетки слоя k) — тот же шаг, что в DI (`value.py`), но поток RK4 (`gcell.rk4`). Агент — Q-жадный: u = argmin_k V(flow(x, u_k, τ_a))."""
import numpy as np
from .gcell import rk4, pendulum


class PendAtlas:
    def __init__(self, n_th=72, n_w=61, wmax=4.0, tau=0.3, umax=0.5, R_goal=0.5, g_over_l=1.0):
        self.f = pendulum(g_over_l); self.umax, self.tau = umax, tau
        self.th = np.arange(n_th) * 2 * np.pi / n_th - np.pi                 # θ ∈ [−π, π), цель у ±π
        self.w = np.linspace(-wmax, wmax, n_w); self.dth = 2 * np.pi / n_th; self.dw = self.w[1] - self.w[0]
        TH, W = np.meshgrid(self.th, self.w, indexing='ij')
        self.goal = np.hypot(((TH % (2 * np.pi)) - np.pi), W) < R_goal       # расстояние до (π, 0) по кругу
        X = np.stack([TH, W], -1)
        self.ends = [rk4(self.f, X, s * umax, tau) for s in (+1, -1)]
        self.V = np.full(TH.shape, 1e3); self.V[self.goal] = 0.0

    def interp(self, V, P):
        """Билинейно, θ периодично, ω вне сетки — BIG. P: (..., 2)."""
        fth = ((P[..., 0] + np.pi) / self.dth) % V.shape[0]; fw = (P[..., 1] - self.w[0]) / self.dw
        inside = (fw >= 0) & (fw <= len(self.w) - 1); fw = np.clip(fw, 0, len(self.w) - 1 - 1e-9)
        i = fth.astype(int) % V.shape[0]; i1 = (i + 1) % V.shape[0]; u = fth - np.floor(fth); j = fw.astype(int); w = fw - j
        r = (1 - u) * (1 - w) * V[i, j] + u * (1 - w) * V[i1, j] + (1 - u) * w * V[i, j + 1] + u * w * V[i1, j + 1]
        return np.where(inside, r, 1e3)

    def solve(self, iters=400, tol=1e-9):
        for it in range(iters):
            Vn = np.minimum(self.V, np.minimum(*[self.tau + self.interp(self.V, e) for e in self.ends])); Vn[self.goal] = 0.0
            ch = float(np.max(self.V - Vn)); self.V = Vn
            if ch < tol:
                break
        return it + 1

    def value(self, x):
        return float(self.interp(self.V, np.asarray(x, float)))

    def in_goal(self, x, R=0.5):
        return np.hypot(((x[0] % (2 * np.pi)) - np.pi), x[1]) < R

    def run_agent(self, x, dt=0.02, T_max=60.0, R=0.5):
        """Q-жадный агент: раз в такт выбирает u_k, минимизирующее V(flow(x, u_k, τ)). Возвращает (время до цели или None, число переключений)."""
        x = np.array(x, float); t = 0.0; sw = 0; up = 0.0
        while t < T_max:
            if self.in_goal(x, R):
                return t, sw
            q = [self.value(rk4(self.f, x, s * self.umax, self.tau)) for s in (+1, -1)]
            u = self.umax * (+1 if q[0] <= q[1] else -1)
            if u != up:
                sw += 1; up = u
            x = rk4(self.f, x, u, dt); t += dt
        return None, sw
