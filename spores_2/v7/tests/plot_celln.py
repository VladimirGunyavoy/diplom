"""Картинки клеток nD: проекция на первые две координаты (дифдрайв — x,y; манипулятор — q1,q2): центральные пути и точки спор. python3 tests/plot_celln.py <система> [N]"""
import sys; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.cells7.systemsn import SYSTEMS
from src.cells7.covern import cover_layer_n
name = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 150; S = SYSTEMS[name](); fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for k in (0, 1):
    cells, P, cov, _ = cover_layer_n(S, k, 2000, N, 400)
    for C in cells:
        L = C.phi(np.zeros((9, C.m)), np.linspace(0, C.tau, 9)); ax[k].plot(L[:, 0], L[:, 1], '-', lw=0.8); ax[k].plot(*C.c[:2], 'k.', ms=2)
    ax[k].set_title('%s слой %d: %d клеток, проекция на (%s)' % (name, k, len(cells), 'x,y' if name.startswith('dd') else 'q1,q2'), fontsize=9)
plt.tight_layout(); plt.savefig('reports/figures/celln_%s.png' % name, dpi=90)
