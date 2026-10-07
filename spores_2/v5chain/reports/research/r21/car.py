"""research-21: «статик кар» (`knowledge/research/static_car.md`) — API коридора v6: flow(P, s, t), clearance(P). Слои — 4 вершины (a, ω) = (±1.5, ±1.8)."""
import math, numpy as np
K, AM, WM, VL, VH, XL = .625, 1.5, 1.8, -1., 2., 5.
HZ = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); HR = .70; GOAL, GR = np.array([4., 3.6]), .45
import os
US = np.array([[a, w] for a in (-AM, AM) for w in ((-WM, 0., WM) if int(os.environ.get('CARL', 4)) >= 6 else (-WM, WM))] + ([[0., -WM], [0., 0.], [0., WM]] if int(os.environ.get('CARL', 4)) >= 9 else []))   # CARL 6: + ω = 0 (прямо), 9: + a = 0 (спектр, concepts)
def f(x, u): return np.stack([x[..., 3] * np.cos(x[..., 2]), x[..., 3] * np.sin(x[..., 2]), u[..., 1] + 0 * x[..., 0], u[..., 0] - K * x[..., 3]], -1)
def _flow_np(P, s, t, dt_max):
    n = max(1, int(np.ceil(abs(t) / dt_max))); h = t / n; u = US[s]; x = np.asarray(P, float)
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
def _flow_scalar(x, s, t, dt_max):
    X, Y, th, v = (float(c) for c in x); a, w = float(US[s][0]), float(US[s][1]); n = max(1, int(math.ceil(abs(t) / dt_max))); h = t / n
    for _ in range(n):
        k1 = (v * math.cos(th), v * math.sin(th), w, a - K * v); t2, v2 = th + h / 2 * k1[2], v + h / 2 * k1[3]
        k2 = (v2 * math.cos(t2), v2 * math.sin(t2), w, a - K * v2); t3, v3 = th + h / 2 * k2[2], v + h / 2 * k2[3]
        k3 = (v3 * math.cos(t3), v3 * math.sin(t3), w, a - K * v3); t4, v4 = th + h * k3[2], v + h * k3[3]
        k4 = (v4 * math.cos(t4), v4 * math.sin(t4), w, a - K * v4)
        X += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); Y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); th += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]); v += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    return np.array([X, Y, th, v])
def flow(P, s, t, dt_max=.05):
    P = np.asarray(P, float); return _flow_scalar(P, s, t, dt_max) if P.ndim == 1 else _flow_np(P, s, t, dt_max)
def clearance(P):
    """≥ 0 — допустимо: до дисков, до стен поля, v в [VL, VH] (по последней оси — минимум)"""
    P = np.asarray(P, float); dh = (np.linalg.norm(P[..., None, :2] - HZ, axis=-1) - HR).min(-1); dw = XL - np.abs(P[..., :2]).max(-1)
    return np.minimum(np.minimum(dh, dw), np.minimum(P[..., 3] - VL, VH - P[..., 3]))
