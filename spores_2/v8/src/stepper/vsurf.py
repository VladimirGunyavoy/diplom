"""Поверхность V (v8, ДИ 2D; слово пользователя 2026-10-12): высота = V·VH, цвет — многоцветный полупрозрачный градиент по V, узлы клеток — точки на поверхности, скелет — рёбра между узлами.
V считает solve8.Graph (E1 вдоль потока, E2 вход в цель, E3 переход между слоями) на ТЕКУЩЕМ атласе (готовые клетки показанных слоёв); ~0.1–0.9 с на три слоя.
Чистый numpy (тестируется без Ursina): solve_layers → build_mesh. Рисует GrowView (одним мешем на слой и тип).
V = ∞ (узел не достигает цели в графе): дыра в поверхности (треугольники с такими вершинами не строятся), узел — серая точка на высоте 0.
Общий слой V*: win — маска «узел выигрывает» (его V ≤ min по слоям V_u(x узла) + eps, при равенстве — слой с меньшим номером); оставляем только рёбра/треугольники/узлы, все вершины которых выигрывают.
Столбцы, входящие в цель (goalcols): есть узел внутри круга RHO или конечный solve8.entry_time на шаге столбца (та же логика, что E2 в Graph.prepare)."""
import os, time
from types import SimpleNamespace
import numpy as np

VH = float(os.environ.get('VSURF_H', .35))                 # высота на единицу V (в мировых единицах до зума)
ALPHA = float(os.environ.get('VSURF_ALPHA', .55))
LIFT = float(os.environ.get('VSURF_LIFT', .04))            # подъём скелета и узлов над поверхностью (мировые единицы высоты), чтобы линии не тонули в радуге
LONG_A = float(os.environ.get('VSURF_LONG_ALPHA', .6)); LONG_W = int(os.environ.get('VSURF_LONG_W', 2))        # боковые линии вдоль траекторий эволюции (столбцы j ≠ центрального), цвет слоя
CENTER_A = float(os.environ.get('VSURF_CENTER_ALPHA', 1.)); CENTER_W = int(os.environ.get('VSURF_CENTER_W', 4))   # центральная линия эволюции клетки (столбец M//2): непрозрачная и жирнее
CROSS_A = float(os.environ.get('VSURF_CROSS_ALPHA', .4)); CROSS_W = int(os.environ.get('VSURF_CROSS_W', 1))      # поперечные линии (узлы одной строки)
GOAL_A = float(os.environ.get('VSURF_GOAL_ALPHA', 1.)); GOAL_W = int(os.environ.get('VSURF_GOAL_W', 6))          # траектории, входящие в цель: белые, жирнее центральных, поверх всего
WIN_EPS = float(os.environ.get('VSURF_WIN_EPS', 1e-3))
BIG = 1e9                                                   # solve8.BIG: V ≥ BIG/2 — «нет пути»
GRAD = np.array([[0., .20, .25, .95], [.25, .10, .75, .95], [.5, .20, .90, .30], [.75, 1., .85, .10], [1., .95, .20, .15]])     # t, R, G, B: синий → бирюзовый → зелёный → жёлтый → красный


def gradient(t, alpha=ALPHA):
    """t ∈ [0, 1] (любой формы) → RGBA (..., 4) по кусочно-линейному градиенту GRAD"""
    t = np.clip(np.asarray(t, float), 0., 1.); out = np.empty(t.shape + (4,))
    for k in range(3): out[..., k] = np.interp(t, GRAD[:, 0], GRAD[:, k + 1])
    out[..., 3] = alpha; return out


def solve_layers(cells_by_u, g=None):
    """cells_by_u: {u: [cell_dict, ...]} (порядок слоёв — порядок ключей) → ({u: [V (nt, M) на клетку]}, info). V ≥ BIG/2 — нет пути.
    info: n, t, passes, finite, vmax, layers, wins {u: [bool (nt, M)]} — узел выигрывает у остальных слоёв, goalcols {u: [bool (M,)]} — столбец входит в цель."""
    from ..algo import solve8
    g = g or solve8.load_growN()
    atlas = {u: [SimpleNamespace(G=np.asarray(c['G']), tau=np.asarray(c['tau']), r=np.asarray(c['r']), hb=int(c['hb']), nb=int(c['nb']), nf=int(c['nf'])) for c in cl] for u, cl in cells_by_u.items() if len(cl)}
    if not atlas: return {}, dict(n=0, t=0., passes=0, wins={}, goalcols={})
    t0 = time.perf_counter(); G = solve8.make_graph(g, atlas); G.solve(verbose=False); t = time.perf_counter() - t0; V = G.V; out = {}; off = 0
    Vl = G.Vlayers(G.P); Vl[np.arange(G.n), G.layer] = np.where(V < BIG / 2, V, np.inf); best = Vl.min(1)                  # свой слой — V узла (не интерполяция); победитель — первый слой в пределах eps от минимума
    wall = (np.argmax(Vl <= best[:, None] + WIN_EPS, axis=1) == G.layer) & np.isfinite(best); wins = {}; gcols = {}; rho = G.rho
    for u, cl in atlas.items():
        out[u] = []; wins[u] = []; gcols[u] = []
        for c in cl:
            nt, M = c.G.shape[:2]; out[u].append(V[off:off + nt * M].reshape(nt, M).copy()); wins[u].append(wall[off:off + nt * M].reshape(nt, M).copy()); off += nt * M
            inn = np.linalg.norm(c.G, axis=-1) <= rho + 1e-12; cost = c.tau[1:] - c.tau[:-1]; m = (~inn[:-1] & (cost > 1e-12)).ravel(); te = np.full(len(m), np.inf)
            if m.any(): te[m] = solve8.entry_time(c.G[:-1].reshape(-1, 2)[m], u, cost.ravel()[m], rho)
            gcols[u].append(inn.any(0) | np.isfinite(te.reshape(nt - 1, M)).any(0))
    assert off == G.n
    fin = V < BIG / 2
    return out, dict(n=int(G.n), t=t, passes=int(G.passes), finite=float(fin.mean()), vmax=float(V[fin].max()) if fin.any() else 0., layers={u: len(c) for u, c in atlas.items()}, wins=wins, goalcols=gcols)


def vscale(Vs_u, q=99.):
    """верх шкалы цвета: q-й процентиль конечных V по всем показанным слоям (хвост не убивает градиент)"""
    a = np.concatenate([v[v < BIG / 2].ravel() for vs in Vs_u.values() for v in vs] or [np.zeros(0)]); return float(np.percentile(a, q)) if len(a) else 1.


def build_mesh(cells, Vs, ax=(0, 1), vmax=1., h=VH, alpha=ALPHA, y0=.01, win=None, goalcols=None, lift=LIFT):
    """клетки слоя + их V (nt, M) → dict: verts (K, 3), colors (K, 4), tris (T,) индексы, pts (P, 3), pcol (P, 4) — узлы на поверхности, inf (Q, 3) — узлы без V (серые, y = y0),
    long (2E, 3) — рёбра вдоль потока (k,j)–(k+1,j), j ≠ M//2, парами вершин, center (2E', 3) — то же для центрального столбца j = M//2, cross (2F, 3) — рёбра поперёк (k,j)–(k,j+1),
    goal (2E'', 3) — рёбра столбцов, входящих в цель (goalcols[i] — bool (M,)); ребро с V=∞ нет. win — bool (nt, M) на клетку: оставить только выигрывающие узлы (общий слой V*; серых ∞-узлов тогда нет).
    Координаты Ursina: x = ось ax[0], y = V·h (вверх), z = ось ax[1]; скелет и узлы — на lift выше."""
    VV, CC, TT, PP, PC, II, LG, CR, CE, GL = [], [], [], [], [], [], [], [], [], []; n0 = 0
    for i, (c, V) in enumerate(zip(cells, Vs)):
        G = np.asarray(c['G']); nt, M = V.shape; fin = V < BIG / 2
        if win is not None: fin = fin & win[i]
        P = np.empty((nt, M, 3)); P[..., 0] = G[..., ax[0]]; P[..., 2] = G[..., ax[1]]; P[..., 1] = np.where(fin, V * h, y0)
        col = gradient(np.where(fin, V, 0.) / max(vmax, 1e-9), alpha); col[~fin, 3] = 0.
        ids = n0 + np.arange(nt * M).reshape(nt, M); a, b, c_, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]; fa, fb, fc, fd = fin[:-1, :-1], fin[:-1, 1:], fin[1:, :-1], fin[1:, 1:]
        t1 = np.stack([a, b, c_], -1)[fa & fb & fc]; t2 = np.stack([b, d, c_], -1)[fb & fd & fc]; TT.append(np.concatenate([t1, t2]).ravel())
        VV.append(P.reshape(-1, 3)); CC.append(col.reshape(-1, 4)); n0 += nt * M
        Pl = P + [0., lift, 0.]; okl = fin[:-1] & fin[1:]; okc = fin[:, :-1] & fin[:, 1:]                                 # скелет на поверхности (+lift): ребро с вершиной V=∞ не рисуется
        cmask = np.zeros(M, bool); cmask[M // 2] = True; gm = np.zeros(M, bool) if goalcols is None else np.asarray(goalcols[i], bool)
        e2 = lambda ok: np.stack([Pl[:-1][ok], Pl[1:][ok]], 1).reshape(-1, 3)
        GL.append(e2(okl & gm[None, :])); CE.append(e2(okl & cmask[None, :])); LG.append(e2(okl & ~cmask[None, :])); CR.append(np.stack([Pl[:, :-1][okc], Pl[:, 1:][okc]], 1).reshape(-1, 3))
        PP.append(P[fin] + [0., lift, 0.]); PC.append(np.broadcast_to([.96, .96, .96, .95], (int(fin.sum()), 4)))
        if win is None: II.append(P[~fin])
    z3, z4 = np.zeros((0, 3)), np.zeros((0, 4)); cat = lambda L, z: np.concatenate(L) if L else z
    return dict(verts=cat(VV, z3), colors=cat(CC, z4), tris=cat(TT, np.zeros(0, int)).astype(np.uint32), pts=cat(PP, z3), pcol=cat(PC, z4), inf=cat(II, z3), long=cat(LG, z3), center=cat(CE, z3), cross=cat(CR, z3), goal=cat(GL, z3))
