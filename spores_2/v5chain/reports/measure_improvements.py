"""Замеры АТЛАС §7 (PLAN п.4–8) -> reports/improvements.json. Запуск: python3 reports/measure_improvements.py"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas import T_star
from src.atlas.improve import Atlas, make_grid, sample_points, metrics, refine

R = {}
pts = sample_points()
C = make_grid(-4, 4, 81); D = C.copy()
at = Atlas(C, D)
node_err = lambda a_, ref: max(abs(t - ref(*a_.nodes[k])) for k, t in a_.T.items() if abs(k[2]) != 2)   # узлы решётки; вспомогательные узлы линий v=±vmax — отдельно
line_err = lambda a_, ref: max([abs(t - ref(*a_.nodes[k])) for k, t in a_.T.items() if abs(k[2]) == 2] or [0.0])

# п.4 складка v=0 + п.5 излом
R["base_nodes"] = len(at.nodes)
R["p4_p5"] = {
    "bilinear (базовая)": metrics(at.bilinear, T_star, pts),
    "hybrid: (c,v) только где билинейная не определена": metrics(at.hybrid, T_star, pts),
    "hybrid: (c,v) при |v|<0.3": metrics(lambda x, v: at.hybrid(x, v, 0.3), T_star, pts),
    "hybrid: (c,v) при |v|<0.6": metrics(lambda x, v: at.hybrid(x, v, 0.6), T_star, pts),
    "bellman-min (излом) поверх hybrid": metrics(at.bellman, T_star, pts),
    "bellman-min поверх hybrid(0.3)": metrics(lambda x, v: at.bellman(x, v, lambda a, b: at.hybrid(a, b, 0.3)), T_star, pts),
}
# п.4б сгущение: равномерное измельчение сетки как цена (сгущение d−c=(kΔ)² на тензорной сетке невозможно — см. отчёт)
R["p4_uniform"] = {}
for n in (41, 81, 121, 161):
    g = make_grid(-4, 4, n); a2 = Atlas(g, g.copy())
    R["p4_uniform"][n] = dict(nodes=len(a2.nodes), bilinear=metrics(a2.bilinear, T_star, pts), hybrid03=metrics(lambda x, v: a2.hybrid(x, v, 0.3), T_star, pts))

# п.6 адаптивные споры (индикатор — T* в центрах ячеек; тензорная вставка середин)
R["p6_adaptive"] = []
for tol in (0.05, 0.02):
    Ca = make_grid(-4, 4, 41); Da = Ca.copy(); hist = []
    for it in range(4):
        a2 = Atlas(Ca, Da)
        hist.append(dict(iter=it, C=len(Ca), D=len(Da), nodes=len(a2.nodes), node_err=node_err(a2, T_star), hybrid03=metrics(lambda x, v: a2.hybrid(x, v, 0.3), T_star, pts)))
        Ca, Da, k = refine(Ca, Da, T_star, tol)
        if k == 0:
            break
    R["p6_adaptive"].append(dict(tol=tol, hist=hist))

# п.7 |v|<=vmax
R["p7_vmax"] = {}
for vm in (1.0, 0.5):
    a2 = Atlas(C, D, vmax=vm)
    ref = lambda x, v, vm=vm: T_star(x, v, vmax=vm)
    ptv = [(x, v) for x, v in pts if abs(v) <= 0.8 * vm]
    R["p7_vmax"][vm] = dict(nodes=len(a2.nodes), reached=len(a2.T), node_err=node_err(a2, ref), line_node_err=line_err(a2, ref),
                            hybrid03=metrics(lambda x, v: a2.hybrid(x, v, 0.3), ref, ptv),
                            no_saturation_ref_gap=max(abs(ref(x, v) - T_star(x, v)) for x, v in ptv))

# п.8 цель-множество: отрезок v=0, x∈[xa,xb]
xa, xb = -0.53, 0.47
xg = np.linspace(xa, xb, 401)
ref_set = lambda x, v: min(T_star(x - g, v) for g in xg)
R["p8_goalset"] = {}
for name, extra in (("без парабол границы", []), ("с параболами границы", [xa, xb])):
    g = make_grid(-4, 4, 81, extra)
    goals = [(i, i, 0) for i in range(len(g)) if xa <= g[i] <= xb]
    a2 = Atlas(g, g.copy(), goals=goals)
    R["p8_goalset"][name] = dict(nodes=len(a2.nodes), reached=len(a2.T), goal_nodes=len(goals),
                                 node_err=node_err(a2, ref_set), hybrid03=metrics(lambda x, v: a2.hybrid(x, v, 0.3), ref_set, pts[:600]))

json.dump(R, open(os.path.join(os.path.dirname(__file__), "improvements.json"), "w"), indent=1, ensure_ascii=False)
print("done")
