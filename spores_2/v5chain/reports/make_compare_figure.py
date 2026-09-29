"""Ошибка интерполяции vs число узлов: равномерная сетка и адаптивное дробление (из improvements.json)."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(__file__); R = json.load(open(f"{d}/improvements.json"))
fig, ax = plt.subplots(figsize=(6, 4.5))
u = R["p4_uniform"]; ns = [u[k]["nodes"] for k in u]
ax.loglog(ns, [u[k]["hybrid03"]["mean"] for k in u], "o-", label="равномерная, hybrid(0.3)")
ax.loglog(ns, [u[k]["bilinear"]["mean"] for k in u], "s--", label="равномерная, билинейная (только покрытые)")
h = R["p6_adaptive"][0]["hist"]
ax.loglog([x["nodes"] for x in h], [x["hybrid03"]["mean"] for x in h], "^-", label="адаптивная (из 41×41), hybrid(0.3)")
ax.set(xlabel="узлов", ylabel="mean |T_interp − T*|", title="Цена/точность"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.savefig(f"{d}/figures/error_vs_nodes.png", dpi=120, bbox_inches="tight")
