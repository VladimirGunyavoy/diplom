"""research-24: сечения ⟂ потоку у дифдрайва (слой u = (v, ω)). Проверки:
(1) голономия «шаг по базису → пересчёт базиса → шаг»: порядок (a e1, b e2) против (b e2, a e1) — сдвиг вдоль f (в единицах времени);
(2) радиальное сечение (шаг w = a e1 + b e2 мелкими подшагами с проекцией направления на f⊥ в текущей точке) — сдвиг от плоского сечения;
(3) плоское сечение ⟂ f(зерно): min cos угла между f(x) и нормалью n0 по сечению полуширины r (трансверсальность)."""
import numpy as np
def f(p, u): v, w = u; return np.array([v * np.cos(p[2]), v * np.sin(p[2]), w])
def P(p, u): fh = f(p, u) / np.linalg.norm(f(p, u)); return np.eye(3) - np.outer(fh, fh)
def basis0(p, u):
    out = []
    for v in np.eye(3)[[2, 0, 1]] if u[0] else np.eye(3):     # θ первым у слоя с v ≠ 0 (оси growN при J ≠ 0 — те же с точностью до знака)
        v = P(p, u) @ v
        for e in out: v = v - (v @ e) * e
        if np.linalg.norm(v) > 1e-6: out.append(v / np.linalg.norm(v))
        if len(out) == 2: break
    return np.array(out)
def transport(E, p, u):                                       # перенос базиса в новую точку: проекция на f⊥ + Грам–Шмидт (RMF первого порядка)
    out = []
    for v in E:
        v = P(p, u) @ v
        for e in out: v = v - (v @ e) * e
        out.append(v / np.linalg.norm(v))
    return np.array(out)
def steps(p, u, E, seq, n=200):                               # seq: [(k, длина)], шаг по оси k с пересчётом базиса на каждом подшаге
    for k, L in seq:
        for _ in range(n): p = p + (L / n) * E[k]; E = transport(E, p, u)
    return p
def radial(p, u, E, a, b, n=400):
    w = a * E[0] + b * E[1]; L = np.hypot(a, b); d = w / L
    for _ in range(n): p = p + (L / n) * d; d = P(p, u) @ d; d /= np.linalg.norm(d)
    return p
for u in [(1., 0.), (.5, .5), (0., 1.)]:
    p0 = np.array([0., 0., .3]); E = basis0(p0, u); fh = f(p0, u) / np.linalg.norm(f(p0, u)); sp = np.linalg.norm(f(p0, u))
    print('u =', u, ' ω∧dω = −v² =', -u[0] ** 2)
    for r in [.05, .1, .2, .5]:
        A = steps(p0, u, E, [(0, r), (1, r)]); B = steps(p0, u, E, [(1, r), (0, r)])
        flat = p0 + r * E[0] + r * E[1]; R = radial(p0, u, E, r, r)
        cosmin = min(abs(f(p0 + s * E[0] + t * E[1], u) @ fh) / sp for s in np.linspace(-r, r, 21) for t in np.linspace(-r, r, 21))
        print(f'  r {r:4}: голономия |A−B|·f̂ = {abs((A - B) @ fh) / sp:.4f} (в ед. времени; r² = {r*r:.4f}), |A−B| = {np.linalg.norm(A - B):.4f};'
              f' радиальное − плоское = {np.linalg.norm(R - flat):.4f}; min cos(f, n0) по плоскому = {cosmin:.3f}')
