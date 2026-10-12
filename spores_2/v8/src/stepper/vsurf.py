"""Поверхность V (v8, ДИ 2D; слово пользователя 2026-10-12): высота = V·VH, цвет — многоцветный полупрозрачный градиент по V, узлы клеток — точки на поверхности.
V считает solve8.Graph (E1 вдоль потока, E2 вход в цель, E3 переход между слоями) на ТЕКУЩЕМ атласе (готовые клетки показанных слоёв); ~0.1 с на три слоя.
Чистый numpy (тестируется без Ursina): solve_layers → build_mesh. Рисует GrowView (одним мешем на слой, как линии).
V = ∞ (узел не достигает цели в графе): дыра в поверхности (треугольники с такими вершинами не строятся), узел — серая точка на высоте 0."""
import os, time
from types import SimpleNamespace
import numpy as np

VH = float(os.environ.get('VSURF_H', .35))                 # высота на единицу V (в мировых единицах до зума)
ALPHA = float(os.environ.get('VSURF_ALPHA', .55))
LONG_A = float(os.environ.get('VSURF_LONG_ALPHA', .9))     # скелет: линии вдоль траекторий эволюции (узлы одного столбца)
CROSS_A = float(os.environ.get('VSURF_CROSS_ALPHA', .25))   # скелет: поперечные линии (узлы одной строки), заметно прозрачнее
BIG = 1e9                                                   # solve8.BIG: V ≥ BIG/2 — «нет пути»
GRAD = np.array([[0., .20, .25, .95], [.25, .10, .75, .95], [.5, .20, .90, .30], [.75, 1., .85, .10], [1., .95, .20, .15]])     # t, R, G, B: синий → бирюзовый → зелёный → жёлтый → красный


def gradient(t, alpha=ALPHA):
    """t ∈ [0, 1] (любой формы) → RGBA (..., 4) по кусочно-линейному градиенту GRAD"""
    t = np.clip(np.asarray(t, float), 0., 1.); out = np.empty(t.shape + (4,))
    for k in range(3): out[..., k] = np.interp(t, GRAD[:, 0], GRAD[:, k + 1])
    out[..., 3] = alpha; return out


def solve_layers(cells_by_u, g=None):
    """cells_by_u: {u: [cell_dict, ...]} (порядок слоёв — порядок ключей) → ({u: [V (nt, M) на клетку]}, info). V ≥ BIG/2 — нет пути."""
    from ..algo import solve8
    g = g or solve8.load_growN()
    atlas = {u: [SimpleNamespace(G=np.asarray(c['G']), tau=np.asarray(c['tau']), r=np.asarray(c['r']), hb=int(c['hb']), nb=int(c['nb']), nf=int(c['nf'])) for c in cl] for u, cl in cells_by_u.items() if len(cl)}
    if not atlas: return {}, dict(n=0, t=0., passes=0)
    t0 = time.perf_counter(); G = solve8.make_graph(g, atlas); G.solve(verbose=False); t = time.perf_counter() - t0; V = G.V; out = {}; off = 0
    for u, cl in atlas.items():
        out[u] = []
        for c in cl: nt, M = c.G.shape[:2]; out[u].append(V[off:off + nt * M].reshape(nt, M).copy()); off += nt * M
    assert off == G.n
    fin = V < BIG / 2
    return out, dict(n=int(G.n), t=t, passes=int(G.passes), finite=float(fin.mean()), vmax=float(V[fin].max()) if fin.any() else 0., layers={u: len(c) for u, c in atlas.items()})


def vscale(Vs_u, q=99.):
    """верх шкалы цвета: q-й процентиль конечных V по всем показанным слоям (хвост не убивает градиент)"""
    a = np.concatenate([v[v < BIG / 2].ravel() for vs in Vs_u.values() for v in vs] or [np.zeros(0)]); return float(np.percentile(a, q)) if len(a) else 1.


def build_mesh(cells, Vs, ax=(0, 1), vmax=1., h=VH, alpha=ALPHA, y0=.01):
    """клетки слоя + их V (nt, M) → dict: verts (K, 3), colors (K, 4), tris (T,) индексы, pts (P, 3), pcol (P, 4) — узлы на поверхности, inf (Q, 3) — узлы без V (серые, y = y0), long (2E, 3) — рёбра вдоль потока (k,j)–(k+1,j) парами вершин, cross (2F, 3) — рёбра поперёк (k,j)–(k,j+1); ребро с V=∞ нет.
    Координаты Ursina: x = ось ax[0], y = V·h (вверх), z = ось ax[1]."""
    VV, CC, TT, PP, PC, II, LG, CR = [], [], [], [], [], [], [], []; n0 = 0
    for c, V in zip(cells, Vs):
        G = np.asarray(c['G']); nt, M = V.shape; fin = V < BIG / 2; P = np.empty((nt, M, 3)); P[..., 0] = G[..., ax[0]]; P[..., 2] = G[..., ax[1]]; P[..., 1] = np.where(fin, V * h, y0)
        col = gradient(np.where(fin, V, 0.) / max(vmax, 1e-9), alpha); col[~fin, 3] = 0.
        ids = n0 + np.arange(nt * M).reshape(nt, M); a, b, c_, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]; fa, fb, fc, fd = fin[:-1, :-1], fin[:-1, 1:], fin[1:, :-1], fin[1:, 1:]
        t1 = np.stack([a, b, c_], -1)[fa & fb & fc]; t2 = np.stack([b, d, c_], -1)[fb & fd & fc]; TT.append(np.concatenate([t1, t2]).ravel())
        VV.append(P.reshape(-1, 3)); CC.append(col.reshape(-1, 4)); n0 += nt * M
        Pl = P + [0., .004, 0.]; okl = fin[:-1] & fin[1:]; okc = fin[:, :-1] & fin[:, 1:]                       # скелет на поверхности: ребро с вершиной V=∞ не рисуется
        LG.append(np.stack([Pl[:-1][okl], Pl[1:][okl]], 1).reshape(-1, 3)); CR.append(np.stack([Pl[:, :-1][okc], Pl[:, 1:][okc]], 1).reshape(-1, 3))
        PP.append(P[fin] + [0., .004, 0.]); PC.append(np.broadcast_to([.96, .96, .96, .95], (int(fin.sum()), 4))); II.append(P[~fin])
    z3, z4 = np.zeros((0, 3)), np.zeros((0, 4)); cat = lambda L, z: np.concatenate(L) if L else z
    return dict(verts=cat(VV, z3), colors=cat(CC, z4), tris=cat(TT, np.zeros(0, int)).astype(np.uint32), pts=cat(PP, z3), pcol=cat(PC, z4), inf=cat(II, z3), long=cat(LG, z3), cross=cat(CR, z3))
