"""Дыры покрытия бабочек маятника: V=BIG на сетке (φ,ω) окна; заполнение спорами (hub-worker-14). Из spores_2/v7. HOLEFILL=1 — добавить споры в дырах и пересчитать V."""
import sys, os, time; sys.path.insert(0, '.'); sys.argv = ['x']
import numpy as np
from src.cells7.butterfly_pend import *
t0 = time.time(); lg = lambda *a: print(*a, round(time.time() - t0), flush=True)
B = ButterflyPend(N=5000, m=7, tau=2.0, win=.3); B.V = np.load('/tmp/claude-1000/bp_V4.npy')
G = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, 91), np.linspace(-3.5, 3.5, 71), indexing='ij'), -1).reshape(-1, 2)
bad = np.array([B.best(g)[0] >= BIG / 2 for g in G]); lg('точек сетки', len(G), 'дыр', int(bad.sum()), round(bad.mean(), 4))
H = G[bad]; print('дыры (φ,ω) примеры', H[:10].round(2).tolist()); np.save('/tmp/claude-1000/bp_holes.npy', H)
