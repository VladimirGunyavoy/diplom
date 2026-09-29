"""Картинки базовой версии атласа (PLAN п.3): T, ошибка интерполяции, решётка + кривая переключения."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.atlas import key, build_lattice, build_edges, cost_to_go, interp_T, T_star

OUT = os.path.join(os.path.dirname(__file__), "figures")
C = np.linspace(-4, 4, 81); D = C.copy()
nodes = build_lattice(C, D); edges = build_edges(nodes)
i0 = int(np.argmin(abs(C)))
T, _ = cost_to_go(edges, key(i0, i0, 0, C, D))

L = 1.5
xs = np.linspace(-L, L, 301); vs = np.linspace(-L, L, 301)
Ti = np.array([[interp_T(x, v, C, D, T) for x in xs] for v in vs])
Ts = np.array([[T_star(x, v) for x in xs] for v in vs])
err = np.abs(Ti - Ts)
vv = np.linspace(-L, L, 400)
sw = -vv*np.abs(vv)/2

def sw_line(ax):
    ax.plot(sw, vv, "w--" if ax is not None else "k--", lw=1)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(xs, vs, Ti, shading="auto"); ax.plot(sw, vv, "w--", lw=1)
ax.set(xlabel="x", ylabel="v", title="T (интерполяция атласа)"); fig.colorbar(im)
fig.savefig(f"{OUT}/T_phase.png", dpi=120, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(xs, vs, err, shading="auto"); ax.plot(sw, vv, "w--", lw=1)
ax.set(xlabel="x", ylabel="v", title="|T_interp − T*|"); fig.colorbar(im)
fig.savefig(f"{OUT}/error_map.png", dpi=120, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
xs_n = np.array([p[0] for p in nodes.values()]); vs_n = np.array([p[1] for p in nodes.values()])
ax.plot(xs_n, vs_n, ".", ms=1.5, color="gray")
for c in C[::4]:
    v = np.linspace(-L, L, 200); ax.plot(c + v*v/2, v, lw=.4, color="tab:blue")
for d in D[::4]:
    v = np.linspace(-L, L, 200); ax.plot(d - v*v/2, v, lw=.4, color="tab:red")
ax.plot(sw, vv, "k--", lw=1.2)
ax.set(xlim=(-L, L), ylim=(-L, L), xlabel="x", ylabel="v", title="Споры c (синие), d (красные), узлы, кривая переключения")
fig.savefig(f"{OUT}/lattice_switch.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("ok", np.nanmean(err), np.nanmax(err))
