"""Маятник u = .3: обратное дерево из ОДНОЙ затравки в равновесии (π, 0) вырождается (NB 400 → 7 спор: сдвиг за τ ≈ 0.01 < клетки ρπ ≈ 0.16,
новые споры отбрасываются). Лечение (docking_nd п.1): затравки на границе окна цели gd = Rg (m точек). Замер: размер дерева, покрытие и V/Vfine
проигрыша (replay_value) на 8 запросах check_nd_pend. Запуск из spores_2/v6: python3 ../v5chain/reports/research/pend_back_seeds.py"""
import sys, os, json, time; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import SysN, build_back, replay_value
from src.atlas6.pend import PendAtlas
from src.atlas6.gcell import rk4, pendulum
HERE = os.path.dirname(os.path.abspath(__file__))
u = 0.3; tau = 0.25; Rg = 0.5; f = pendulum(1.0); rho = 0.05
dth = lambda th: (th - np.pi + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: rk4(f, P, u * (1 if s == 0 else -1), t)
gd = lambda P: np.hypot(dth(P[..., 0]), P[..., 1])
F = PendAtlas(n_th=252, n_w=241, wmax=4.0, tau=tau, umax=u, R_goal=Rg); F.solve(iters=1500)
rng = np.random.default_rng(1); Q = []
while len(Q) < 8:
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)])
    if gd(x) > 1.0 and F.value(x) < 40: Q.append(x)
Vf = np.array([F.value(x) for x in Q]); out = []
for m in (0, 16, 32):
    seeds = [(np.pi, 0.0)] if m == 0 else [(np.pi + Rg * 0.999 * np.cos(a), Rg * 0.999 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, m, endpoint=False)]
    S = SysN(fl, 2, (np.pi, np.pi), lambda P: gd(P) < Rg, seeds, per=(2 * np.pi, 0.0), ok=lambda p: abs(p[1]) <= 4.0)
    for NB, NF in ((200, 200), (400, 400)):
        t0 = time.time(); back = build_back(S, tau, NB, rho)
        V = np.array([replay_value(S, tau, x, NB, NF, rho, rho, back=back)[0] for x in Q]); r = V / Vf; ok = np.isfinite(r)
        row = dict(seeds=m, NB=NB, NF=NF, back_size=len(back[0]), found=int(ok.sum()), mean=float(r[ok].mean()) if ok.any() else None,
                   max=float(r[ok].max()) if ok.any() else None, min=float(r[ok].min()) if ok.any() else None, sec=time.time() - t0)
        out.append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, 'pend_back_seeds.json'), 'w'), indent=1)
