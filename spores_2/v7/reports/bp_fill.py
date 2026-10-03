"""Заполнение дыр покрытия бабочек маятника спорами (hub-worker-14): +30 спор вокруг каждой дыры (σ φ .1, ω .25), пересчёт V → /tmp/claude-1000/bp_V5.npy. Из spores_2/v7."""
import sys, time; sys.path.insert(0, '.'); sys.argv = ['x']
import numpy as np
from src.cells7.butterfly_pend import *
t0 = time.time(); lg = lambda *a: print(*a, round(time.time() - t0), flush=True)
H = np.load('/tmp/claude-1000/bp_holes.npy'); rng = np.random.default_rng(5)
extra = np.concatenate([h + rng.normal(0, 1, (30, 2)) * np.array([.1, .25]) for h in H]); extra[:, 0] = wrap(extra[:, 0]); extra[:, 1] = np.clip(extra[:, 1], -4, 4)
B = ButterflyPend(N=5000, m=7, tau=2.0, win=.3, extra=extra); lg('спор', B.K, 'доп.', len(extra))
B.solve(log=lg); np.save('/tmp/claude-1000/bp_V5.npy', B.V); np.save('/tmp/claude-1000/bp_extra.npy', extra); lg('V сохранена, итер', B.n_it, 'пар', B.npairs)
G = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, 91), np.linspace(-3.5, 3.5, 71), indexing='ij'), -1).reshape(-1, 2)
bad = np.array([B.best(g)[0] >= BIG / 2 for g in G]); lg('дыр после заполнения', int(bad.sum()), round(float(bad.mean()), 4))
