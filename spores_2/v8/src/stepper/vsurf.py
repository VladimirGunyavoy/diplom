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
LONG_A = float(os.environ.get('VSURF_LONG_ALPHA', 1.)); LONG_W = int(os.environ.get('VSURF_LONG_W', 2))        # боковые линии вдоль траекторий эволюции (столбцы j ≠ центрального), цвет слоя
CENTER_A = float(os.environ.get('VSURF_CENTER_ALPHA', 1.)); CENTER_W = int(os.environ.get('VSURF_CENTER_W', 4))   # центральная линия эволюции клетки (столбец M//2): непрозрачная и жирнее
CROSS_A = float(os.environ.get('VSURF_CROSS_ALPHA', .8)); CROSS_W = int(os.environ.get('VSURF_CROSS_W', 2))      # поперечные линии (узлы одной строки)
GOAL_A = float(os.environ.get('VSURF_GOAL_ALPHA', 1.)); GOAL_W = int(os.environ.get('VSURF_GOAL_W', 6)); GOAL_SIDE_W = int(os.environ.get('VSURF_GOAL_SIDE_W', 2))   # траектории, входящие в цель: белые, поверх всего; центральный столбец клетки — GOAL_W, остальные входящие — тоньше (GOAL_SIDE_W)
UNDER_A = float(os.environ.get('VSURF_UNDER_ALPHA', 0.)); UNDER_DW = int(os.environ.get('VSURF_UNDER_DW', 2))      # тёмная подложка под боковыми/центральными линиями: ВЫКЛ. по умолчанию (слово пользователя 2026-10-12: чёрные границы не надо), alpha и добавка к толщине
LINE_LIGHT = float(os.environ.get('VSURF_LINE_LIGHT', .4))     # линии скелета на поверхности: оттенок слоя, осветлённый к белому на эту долю (синие слоя читаются на синем градиенте без подложки), alpha 1
def line_color(c, light=None):
    c = np.asarray(c, float)[:3]; l = LINE_LIGHT if light is None else light; return tuple(float(x) for x in c + (1. - c) * l)
WIN_EPS = float(os.environ.get('VSURF_WIN_EPS', 1e-3))
BIG = 1e9                                                   # solve8.BIG: V ≥ BIG/2 — «нет пути»
GRAD = np.array([[0., .20, .25, .95], [.25, .10, .75, .95], [.5, .20, .90, .30], [.75, 1., .85, .10], [1., .95, .20, .15]])     # t, R, G, B: синий → бирюзовый → зелёный → жёлтый → красный


def gradient(t, alpha=ALPHA):
    """t ∈ [0, 1] (любой формы) → RGBA (..., 4) по кусочно-линейному градиенту GRAD"""
    t = np.clip(np.asarray(t, float), 0., 1.); out = np.empty(t.shape + (4,))
    for k in range(3): out[..., k] = np.interp(t, GRAD[:, 0], GRAD[:, k + 1])
    out[..., 3] = alpha; return out


def solve_layers(cells_by_u, g=None, calc='v'):
    """cells_by_u: {u: [cell_dict, ...]} (порядок слоёв — порядок ключей) → ({u: [V (nt, M) на клетку]}, info); calc — 'v' (старый расчёт) | 'q' | 'qquad' (solve8.Graph.set_calc; узлы показывают V = min(Q…), запросы G.Vlayers — по Q). V ≥ BIG/2 — нет пути.
    info: n, t, calc, passes, finite, vmax, layers, wins {u: [bool (nt, M)]} — узел выигрывает у остальных слоёв, goalcols {u: [bool (M,)]} — столбец входит в цель."""
    from ..algo import solve8
    g = g or solve8.load_growN()
    atlas = {u: [SimpleNamespace(G=np.asarray(c['G']), tau=np.asarray(c['tau']), r=np.asarray(c['r']), hb=int(c['hb']), nb=int(c['nb']), nf=int(c['nf'])) for c in cl] for u, cl in cells_by_u.items() if len(cl)}
    if not atlas: return {}, dict(n=0, t=0., passes=0, wins={}, goalcols={})
    t0 = time.perf_counter(); G = solve8.make_graph(g, atlas); G.set_calc(calc); G.solve(verbose=False); t = time.perf_counter() - t0; V = G.Vn; out = {}; off = 0
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
    return out, dict(G=G, calc=calc, n=int(G.n), t=t, passes=int(G.passes), converged=bool(getattr(G, 'converged', False)), finite=float(fin.mean()), vmax=float(V[fin].max()) if fin.any() else 0., layers={u: len(c) for u, c in atlas.items()}, wins=wins, goalcols=gcols)


TRAJ_N = int(os.environ.get('VSURF_TRAJ_N', 300)); TRAJ_SEED = int(os.environ.get('VSURF_TRAJ_SEED', 1)); TRAJ_REC = int(os.environ.get('VSURF_TRAJ_REC', 3))       # итоговые траектории: число стартов, сид, запись каждого REC-го шага (dt=.01)
TRAJ_A = float(os.environ.get('VSURF_TRAJ_ALPHA', 1.)); TRAJ_FW = int(os.environ.get('VSURF_TRAJ_FLOOR_W', 3)); TRAJ_SW = int(os.environ.get('VSURF_TRAJ_SURF_W', 3))


TRAJ_SAT = float(os.environ.get('VSURF_TRAJ_SAT', .62))                              # доля насыщенности цветов траекторий (слово пользователя 2026-10-12: «менее насыщенные»; .62 = −38%, яркость V из HSV остаётся)
def _desat(c, k=TRAJ_SAT):
    import colorsys; h, sa, v = colorsys.rgb_to_hsv(*c); return tuple(float(x) for x in colorsys.hsv_to_rgb(h, sa * k, v))
CTRL_COL = {1: _desat((.72, .25, 1.)), -1: _desat((1., .12, .12)), 0: (.10, .66, .16)}             # цвет управления (слово пользователя 2026-10-12): u>0 фиолетовый, u<0 красный, u=0 ЗЕЛЁНЫЙ (слово пользователя 2026-10-12 ~16:25; тёмнее зелёной затравки и не бирюзовый, как очередь спор); эти два цвета зарезервированы под управление
TRAJ_PT = float(os.environ.get('VSURF_TRAJ_PT', .5))                               # радиус точек траектории — доля радиуса точки споры


def ctrl_color(U, alpha=1.):
    """управления (...,) → RGBA (..., 4): u>0 фиолетовый, u<0 красный, u=0 зелёный"""
    U = np.asarray(U, float); out = np.empty(U.shape + (4,)); out[..., :3] = np.array(CTRL_COL[0]); out[..., :3][U > 1e-9] = CTRL_COL[1]; out[..., :3][U < -1e-9] = CTRL_COL[-1]; out[..., 3] = alpha; return out


def _prephase(G, X, box, dt=.01, rec=TRAJ_REC, maxsteps=3000):
    """старты без V* (решение пользователя 2026-10-12: траектории считать до выхода из области): для каждого слоя u поток из точки шагом dt до первого события «V* определено» (в т.ч. попали в цель) / «вышли за область |x| > box».
    Слой выбирается так: из тех, что приводят в область с V*, — приводящий раньше; если ни один — тот, что дольше остаётся в области. → список dict на старт: P (K, 2), U (K,), t1 (время до V*), X1 (точка, где V* определено, или None), mark (точка выхода на границе или None)"""
    n = len(X); L = G.L; us = np.asarray(G.us, float); S = np.repeat(np.asarray(X, float)[:, None, :], L, 1).copy(); kind = np.zeros((n, L), int); tstop = np.full((n, L), maxsteps); traj = [S.copy()]; box = np.asarray(box, float)
    for step in range(1, maxsteps + 1):
        ii, ll = np.nonzero(kind == 0)
        if not len(ii): break
        x, v, u = S[ii, ll, 0], S[ii, ll, 1], us[ll]; S[ii, ll, 0] = x + v * dt + u * dt * dt / 2; S[ii, ll, 1] = v + u * dt; Y = S[ii, ll]
        out = (np.abs(Y) > box).any(1); kind[ii[out], ll[out]] = 2; tstop[ii[out], ll[out]] = step; k = ~out
        if k.any():
            Vs = G.Vlayers(Y[k]).min(1); d = np.isfinite(Vs) & (Vs < BIG / 2); jj, mm = ii[k][d], ll[k][d]; kind[jj, mm] = 1; tstop[jj, mm] = step
        traj.append(S.copy())
    traj = np.asarray(traj); out_l = []
    for i in range(n):
        dl = np.flatnonzero(kind[i] == 1); l = int(dl[np.argmin(tstop[i, dl])]) if len(dl) else int(np.argmax(tstop[i])); ts = int(tstop[i, l]); P = traj[:ts + 1, i, l]; mark = None; X1 = None
        if kind[i, l] == 1: X1 = P[-1].copy()
        elif kind[i, l] == 2:                                                                    # последняя точка — на границе области (пересечение отрезка последнего шага с рамкой)
            a_, b_ = P[-2], P[-1]; tt = 1.
            for ax in range(2):
                if abs(b_[ax]) > box[ax] and b_[ax] != a_[ax]: tt = min(tt, (np.sign(b_[ax]) * box[ax] - a_[ax]) / (b_[ax] - a_[ax]))
            mark = a_ + max(tt, 0.) * (b_ - a_); P = np.concatenate([P[:-1], mark[None]])
        keep = np.unique(np.r_[np.arange(0, len(P), rec), len(P) - 1]); out_l.append(dict(P=P[keep], U=np.full(len(keep), us[l]), t1=ts * dt if X1 is not None else None, X1=X1, mark=mark))
    return out_l


def agent_paths(G, g=None, n=TRAJ_N, seed=TRAJ_SEED, rec=TRAJ_REC, policy=None):
    """n случайных стартов в поле (|x| ≤ XLV, вне цели, seed фиксирован) → путь агента solve8 на графе G (политика: policy; по умолчанию commit при расчёте по Q, иначе look) для каждого старта.
    У старта без V* путь начинается с _prephase (поток слоя до V* или до выхода за область ±(XL+GM)); вышел — status 'out_of_area' (конец — mark на границе), нашёл V* — дальше обычный агент (может дойти).
    → dict: X0 (n, 2); status (n,) 'reached' | 'failed' | 'out_of_area'; prephase (n,) bool — старт был без V*; left_work (n,) bool — путь хоть раз вышел за ±XL; T (n,) время до цели (с предфазой; nan — не дошёл); sw (n,) переключения агента; mark (n, 2) точка выхода (nan — нет);
    paths [(K, 2)], V [(K,) V*(x) = min по слоям, ∞ вне области V], U [(K,) управление на последнем шаге перед точкой] — по каждому старту (в порядке X0)"""
    from ..algo import solve8
    policy = policy or ('commit' if G.calc in ('q', 'qquad') else 'look'); g = g or solve8.load_growN(); X0 = solve8.starts(g, n, seed); n = len(X0); Vs0 = G.Vstar(X0)[0]; has = np.isfinite(Vs0) & (Vs0 < BIG / 2); box = np.where(g.XLV > 0, g.XLV + g.GM, np.inf)
    pre = {int(i): r for i, r in zip(np.flatnonzero(~has), _prephase(G, X0[~has], box, rec=rec) if (~has).any() else [])}
    src = [i for i in range(n) if has[i] or pre[i]['X1'] is not None]; Xa = np.array([X0[i] if has[i] else pre[i]['X1'] for i in src]); T = np.full(n, np.nan); sw = np.zeros(n, int); P_ = {i: [] for i in range(n)}; U_ = {i: [] for i in range(n)}; mark = np.full((n, 2), np.nan)
    for i, r in pre.items(): P_[i].append(r['P']); U_[i].append(r['U']); mark[i] = r['mark'] if r['mark'] is not None else np.nan
    if src:
        Tr, _ = solve8.t_ref(Xa); path = []; Ta, swa = G.run_agent(Xa, 3 * Tr + 2, rec=rec, path=path, **solve8.POLICIES[policy])
        ids = np.concatenate([a for a, _, _ in path]); P = np.concatenate([x for _, x, _ in path]); Lr = np.concatenate([c for _, _, c in path]); order = np.argsort(ids, kind='stable'); ids, P, Lr = ids[order], P[order], Lr[order]       # внутри агента порядок записи сохранён (stable)
        Ul = np.asarray(G.us, float)[Lr]; cut = np.flatnonzero(np.diff(ids)) + 1; Pp, Up = np.split(P, cut), np.split(Ul, cut)
        for k, i in enumerate(src):
            t1 = pre[i]['t1'] if i in pre else 0.; T[i] = Ta[k] + t1; sw[i] = swa[k]
            if i in pre: P_[i].append(Pp[k][1:]); U_[i].append(Up[k][1:])                      # первая точка агента = конец предфазы
            else: P_[i].append(Pp[k]); U_[i].append(Up[k])
    paths = [np.concatenate(P_[i]) if P_[i] else X0[i][None] for i in range(n)]; Us = [np.concatenate(U_[i]) if U_[i] else np.zeros(1) for i in range(n)]
    allP = np.concatenate(paths); Vst = G.Vlayers(allP).min(1); cut = np.cumsum([len(p) for p in paths])[:-1]; Vl_ = np.split(Vst, cut)
    status = np.where(np.isfinite(T), 'reached', np.where(np.array([i in pre and pre[i]['X1'] is None for i in range(n)]), 'out_of_area', 'failed')).astype(object)
    xlv = np.asarray(g.XLV, float); left = np.array([bool(((np.abs(p) > xlv + 1e-12) & (xlv > 0)).any()) for p in paths])                  # хоть раз вышла за рабочее поле ±XL (без GM)
    return dict(paths=paths, V=Vl_, U=Us, T=T, sw=sw, X0=X0, status=status, prephase=~has, mark=mark, left_work=left)


def traj_geom(res, ax=(0, 1), h=VH, lift=LIFT, y0=.01):
    """траектории → dict: floor (2E, 3) / surf (2E', 3) — пары вершин линий (на полу y = y0, на поверхности V* y = V*·h + lift; сегмент с V* = ∞ — нет), fcol/scol (2E, 4) — цвет управления сегмента, pf/ps (P, 3) и pfc/psc (P, 4) — точки пути (на полу / на поверхности) и их цвет.
    Управление сегмента (k−1 → k) и точки k — то, что действовало на последнем шаге перед k; у точки 0 — как у точки 1."""
    F, FC, S, SC, PF, PFC, PS, PSC = [], [], [], [], [], [], [], []
    for P, V, U in zip(res['paths'], res['V'], res['U']):
        if len(P) < 2: continue
        U = U.copy(); U[0] = U[1]; col = ctrl_color(U); f = np.zeros((len(P), 3)); f[:, 0] = P[:, ax[0]]; f[:, 2] = P[:, ax[1]]; f[:, 1] = y0
        F.append(np.stack([f[:-1], f[1:]], 1).reshape(-1, 3)); FC.append(np.repeat(col[1:], 2, axis=0)); PF.append(f); PFC.append(col)
        ok = np.isfinite(V) & (V < BIG / 2); q = f.copy(); q[:, 1] = np.where(ok, V, 0.) * h + lift; e = ok[:-1] & ok[1:]
        S.append(np.stack([q[:-1][e], q[1:][e]], 1).reshape(-1, 3)); SC.append(np.repeat(col[1:][e], 2, axis=0)); PS.append(q[ok]); PSC.append(col[ok])
    z3, z4 = np.zeros((0, 3)), np.zeros((0, 4)); cat = lambda L, z: np.concatenate(L) if L else z
    mk = np.asarray(res.get('mark', np.zeros((0, 2)))); mk = mk[np.isfinite(mk).all(1)] if len(mk) else mk; M_ = np.zeros((len(mk), 3)); M_[:, 0] = mk[:, ax[0]] if len(mk) else []; M_[:, 2] = mk[:, ax[1]] if len(mk) else []; M_[:, 1] = y0
    return dict(mk=M_, floor=cat(F, z3), fcol=cat(FC, z4), surf=cat(S, z3), scol=cat(SC, z4), pf=cat(PF, z3), pfc=cat(PFC, z4), ps=cat(PS, z3), psc=cat(PSC, z4))


def vscale(Vs_u, q=99.):
    """верх шкалы цвета: q-й процентиль конечных V по всем показанным слоям (хвост не убивает градиент)"""
    a = np.concatenate([v[v < BIG / 2].ravel() for vs in Vs_u.values() for v in vs] or [np.zeros(0)]); return float(np.percentile(a, q)) if len(a) else 1.


def time_nodes(c, V, fin):
    """временной скелет на поверхности V: узлы Gt (те же столбцы-траектории, равные шаги времени tt) → (Gt, Vt (ntt, M), ok (ntt, M)).
    V вдоль столбца j интерполируется линейно по tau[:, j] (тот же приём, что и у столбцов потока); вне диапазона tau и при ∞/не-победителе на любом из двух соседних узлов — нет значения."""
    tau = np.asarray(c['tau'], float); tt = np.asarray(c['tt'], float); nt, M = V.shape; Vt = np.zeros((len(tt), M)); ok = np.zeros((len(tt), M), bool)
    for j in range(M):
        tj = tau[:, j]; i = np.clip(np.searchsorted(tj, tt) - 1, 0, nt - 2); t0, t1 = tj[i], tj[i + 1]; w = np.clip((tt - t0) / np.maximum(t1 - t0, 1e-12), 0., 1.)
        Vt[:, j] = (1 - w) * np.where(fin[i, j], V[i, j], 0.) + w * np.where(fin[i + 1, j], V[i + 1, j], 0.)
        ok[:, j] = fin[i, j] & fin[i + 1, j] & (tt >= tj[0] - 1e-9) & (tt <= tj[-1] + 1e-9)
    return np.asarray(c['Gt']), Vt, ok


def build_mesh(cells, Vs, ax=(0, 1), vmax=1., h=VH, alpha=ALPHA, y0=.01, win=None, goalcols=None, lift=LIFT, skel='normal'):
    """клетки слоя + их V (nt, M) → dict: verts (K, 3), colors (K, 4), tris (T,) индексы, pts (P, 3), pcol (P, 4) — узлы на поверхности, inf (Q, 3) — узлы без V (серые, y = y0),
    long (2E, 3) — рёбра вдоль потока (k,j)–(k+1,j), j ≠ M//2, парами вершин, center (2E', 3) — то же для центрального столбца j = M//2, cross (2F, 3) — рёбра поперёк (k,j)–(k,j+1),
    goal (2E'', 3) — рёбра боковых столбцов (j ≠ M//2), входящих в цель (goalcols[i] — bool (M,)), goalc — то же для центрального столбца; ребро с V=∞ нет. win — bool (nt, M) на клетку: оставить только выигрывающие узлы (общий слой V*; серых ∞-узлов тогда нет).
    skel='time' (клетка с Gt): продольные/поперечные/целевые линии идут по временным узлам Gt на высоте V(t) (интерполяция вдоль столбца по tau); поверхность и узлы — по нормальным G.
    Координаты Ursina: x = ось ax[0], y = V·h (вверх), z = ось ax[1]; скелет и узлы — на lift выше."""
    VV, CC, TT, PP, PC, II, LG, CR, CE, GL, GC = [], [], [], [], [], [], [], [], [], [], []; n0 = 0
    for i, (c, V) in enumerate(zip(cells, Vs)):
        G = np.asarray(c['G']); nt, M = V.shape; fin = V < BIG / 2
        if win is not None: fin = fin & win[i]
        P = np.empty((nt, M, 3)); P[..., 0] = G[..., ax[0]]; P[..., 2] = G[..., ax[1]]; P[..., 1] = np.where(fin, V * h, y0)
        col = gradient(np.where(fin, V, 0.) / max(vmax, 1e-9), alpha); col[~fin, 3] = 0.
        ids = n0 + np.arange(nt * M).reshape(nt, M); a, b, c_, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]; fa, fb, fc, fd = fin[:-1, :-1], fin[:-1, 1:], fin[1:, :-1], fin[1:, 1:]
        t1 = np.stack([a, b, c_], -1)[fa & fb & fc]; t2 = np.stack([b, d, c_], -1)[fb & fd & fc]; TT.append(np.concatenate([t1, t2]).ravel())
        VV.append(P.reshape(-1, 3)); CC.append(col.reshape(-1, 4)); n0 += nt * M
        Pl = P + [0., lift, 0.]; fk = fin
        if skel == 'time' and c.get('Gt') is not None:
            Gt, Vt, fk = time_nodes(c, V, fin); Pl = np.empty(Gt.shape[:2] + (3,)); Pl[..., 0] = Gt[..., ax[0]]; Pl[..., 2] = Gt[..., ax[1]]; Pl[..., 1] = np.where(fk, Vt * h, y0) + lift
        okl = fk[:-1] & fk[1:]; okc = fk[:, :-1] & fk[:, 1:]                                 # скелет на поверхности (+lift): ребро с вершиной V=∞ не рисуется
        cmask = np.zeros(M, bool); cmask[M // 2] = True; gm = np.zeros(M, bool) if goalcols is None else np.asarray(goalcols[i], bool)
        e2 = lambda ok: np.stack([Pl[:-1][ok], Pl[1:][ok]], 1).reshape(-1, 3)
        GL.append(e2(okl & (gm & ~cmask)[None, :])); GC.append(e2(okl & (gm & cmask)[None, :])); CE.append(e2(okl & cmask[None, :])); LG.append(e2(okl & ~cmask[None, :])); CR.append(np.stack([Pl[:, :-1][okc], Pl[:, 1:][okc]], 1).reshape(-1, 3))
        PP.append(P[fin] + [0., lift, 0.]); PC.append(np.broadcast_to([.96, .96, .96, .95], (int(fin.sum()), 4)))
        if win is None: II.append(P[~fin])
    z3, z4 = np.zeros((0, 3)), np.zeros((0, 4)); cat = lambda L, z: np.concatenate(L) if L else z
    return dict(verts=cat(VV, z3), colors=cat(CC, z4), tris=cat(TT, np.zeros(0, int)).astype(np.uint32), pts=cat(PP, z3), pcol=cat(PC, z4), inf=cat(II, z3), long=cat(LG, z3), center=cat(CE, z3), cross=cat(CR, z3), goal=cat(GL, z3), goalc=cat(GC, z3))
