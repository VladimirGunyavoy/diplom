"""
GrowView — рисует снимок паузы алгоритма роста клеток (PLAN п.49, v8).

Контракт снимка (dict, из stepper.pause(**state)):
  phase (str), lvl (int), u (слой), seed (n,), cell {G (rows, M, n), c, r, e, nb, nf} | None,
  cells [cell, ...] (готовые клетки слоя), reason (str | None), queue (k, n).
Плоскость экрана: ось AX[0] -> x, AX[1] -> z (как в v4: position = (x, 0.01, z)).

Геометрия считается чисто на numpy (`build_geometry`) — тестируется без Ursina.
Рисование — по одному Scalable-объекту на слой геометрии (точки/отрезки одним мешем), поэтому
тысячи узлов не создают тысячи Entity.
"""
import os, sys, time, threading
import numpy as np

HALO = .1
SPORE_SCALE = .02                                  # Spore.scale по умолчанию; SporeManager.size в v8 — множитель (1.0), т.к. спор в сцене нет
Y0 = .01


def proj(P, ax):
    """(..., n) -> (..., 3): (x, Y0, z) по осям ax."""
    P = np.asarray(P, float); out = np.empty(P.shape[:-1] + (3,)); out[..., 0] = P[..., ax[0]]; out[..., 1] = Y0; out[..., 2] = P[..., ax[1]]; return out


def cell_rows(G, ax):
    """отрезки строк клетки: (K, 2, 3) — между соседними узлами каждой строки."""
    P = proj(G, ax)                                   # (rows, M, 3)
    return np.stack([P[:, :-1], P[:, 1:]], 2).reshape(-1, 2, 3) if P.shape[1] > 1 else np.zeros((0, 2, 3))


def cell_outline(G, ax, halo=False):
    """контур клетки: ломаная по концам строк, (K, 2, 3). Крайние узлы G стоят на ±(1+HALO)·r (Cell.build) — это край гало (halo=True);
    ядро (halo=False) — те же концы, стянутые к центру строки в 1/(1+HALO) раза."""
    P = proj(G, ax); a, b = P[:, 0].copy(), P[:, -1].copy()
    if not halo:
        c = (a + b) / 2; a, b = c + (a - c) / (1 + HALO), c + (b - c) / (1 + HALO)
    ring = np.concatenate([a, b[::-1]], 0); ring = np.concatenate([ring, ring[:1]], 0)
    return np.stack([ring[:-1], ring[1:]], 1)


def build_strips(snap, ax=(0, 1), skel='normal', cells=None, with_cur=True):
    """линии слоёв как ломаные (по одной на клетку): строки — зигзаг по узлам подряд (L₋₁→C₋₁→R₋₁→L₀→…), контур ядра и гало — замкнутая ломаная; dict имя слоя -> список (n, 3)"""
    out = {k: [] for k in ('segs_rows_done', 'segs_center_done', 'segs_core_done', 'segs_halo_done', 'segs_rows_cur', 'segs_center_cur', 'segs_core_cur', 'segs_halo_cur')}
    def ring(o): return np.concatenate([o[:, 0], o[-1:, 1]], 0)
    def cell(G, suf, c=None):
        hb, hf = (int((c or {}).get('hb') or 0), int((c or {}).get('hf') or 0))      # гало-строки за торцами (growN: hb сзади, hf спереди): ядро — без них, гало — по всем
        Gs = c['Gt'] if skel == 'time' and (c or {}).get('Gt') is not None else G                  # скелет: 'time' — строки/центр по временным узлам Gt, контуры ядра/гало — по нормальному G
        P = proj(Gs, ax)
        if P.shape[1] > 1: out['segs_rows_' + suf].extend(list(P))                        # по полосе на строку (L→C→R): диагонали между строками не рисуются
        if P.shape[0] > 1: out['segs_center_' + suf].append(P[:, P.shape[1] // 2])        # траектория зерна: центральные узлы строк
        Gc = G[hb:len(G) - hf] if (hb or hf) and len(G) - hb - hf >= 1 else G
        out['segs_core_' + suf].append(ring(cell_outline(Gc, ax))); out['segs_halo_' + suf].append(ring(cell_outline(G, ax, True)))
    for c in (snap.get('cells') or []) if cells is None else cells: cell(c['G'], 'done', c)
    cur = snap.get('cell') if with_cur else None
    if cur is not None: cell(cur['G'], 'cur', cur)
    sec = snap.get('section') if with_cur else None
    if sec is not None: out['segs_rows_cur'].append(proj(sec, ax))
    return out


def seg_index(K):
    """индексы отрезков для Mesh(mode='line'): без triangles ursina рисует ОДНУ ломаную через все вершины, а у нас вершины — пары [a0, b0, a1, b1, ...]."""
    return [(2 * i, 2 * i + 1) for i in range(K)]


def build_geometry(snap, ax=(0, 1), halo=HALO, skel='normal'):
    """dict имя слоя -> массив: 'points_*' (K, 3), 'segs_*' (K, 2, 3). Пустые слои — нулевой размер."""
    g = {}; cells = snap.get('cells') or []; cur = snap.get('cell')
    z3, z23 = np.zeros((0, 3)), np.zeros((0, 2, 3))
    def GK(c): return c['Gt'] if skel == 'time' and c.get('Gt') is not None else c['G']
    g['points_done'] = np.concatenate([proj(GK(c), ax).reshape(-1, 3) for c in cells], 0) if cells else z3
    g['segs_rows_done'] = np.concatenate([cell_rows(c['G'], ax) for c in cells], 0) if cells else z23
    g['segs_core_done'] = np.concatenate([cell_outline(c['G'], ax) for c in cells], 0) if cells else z23
    g['segs_halo_done'] = np.concatenate([cell_outline(c['G'], ax, True) for c in cells], 0) if cells else z23
    if cur is not None:
        g['points_cur'] = proj(GK(cur), ax).reshape(-1, 3); g['segs_rows_cur'] = cell_rows(cur['G'], ax)
        g['segs_core_cur'] = cell_outline(cur['G'], ax); g['segs_halo_cur'] = cell_outline(cur['G'], ax, True)
    else: g['points_cur'], g['segs_rows_cur'], g['segs_core_cur'], g['segs_halo_cur'] = z3, z23, z23, z23
    sec = snap.get('section')                                          # пауза section: узлы базового сечения (K, n)
    if sec is not None: g['points_cur'] = np.concatenate([g['points_cur'], proj(sec, ax)], 0); sp = proj(sec, ax); g['segs_rows_cur'] = np.concatenate([g['segs_rows_cur'], np.stack([sp[:-1], sp[1:]], 1)], 0)
    seed = snap.get('seed'); g['points_seed'] = proj(np.asarray(seed, float)[None], ax).reshape(-1, 3) if seed is not None else z3
    q = snap.get('queue'); q = None if q is None else np.asarray(q, float); g['points_queue'] = proj(q.reshape(-1, q.shape[-1]), ax) if q is not None and q.size else z3
    return g


def caption(snap):
    """одна строка подписи: фаза, слой, причина стопа."""
    if snap is None: return 'no snapshot (N - step, G - seed at look point)'
    r = snap.get('reason'); cells = snap.get('cells') or []
    return 'phase: %s  (lvl %s)   layer u=%s   cells: %d   queue: %d%s' % (snap.get('phase'), snap.get('lvl'), snap.get('u'), len(cells), len(snap.get('queue') if snap.get('queue') is not None else []), ('   STOP: %s' % r) if r else '')


class TimeTable:
    """время по именам пауз: вычисления (из stepper.times) и отрисовка (здесь)."""
    def __init__(s): s.draw = {}; s.n = {}
    def add_draw(s, phase, dt): s.draw[phase] = s.draw.get(phase, 0.) + dt; s.n[phase] = s.n.get(phase, 0) + 1
    def text(s, compute=None, top=8):
        compute = compute or {}; names = sorted(set(compute) | set(s.draw), key=lambda k: -(compute.get(k, 0.) + s.draw.get(k, 0.)))[:top]
        rows = ['%-18s %9s %9s %4s' % ('pause', 'compute,ms', 'draw,ms', 'n')] + ['%-18s %9.1f %9.1f %4d' % (k, 1e3 * compute.get(k, 0.), 1e3 * s.draw.get(k, 0.), s.n.get(k, 0)) for k in names]
        return '\n'.join(rows)


def _boxes(P, hw):
    """точки рельса → «вагончики»: K брусьев с полуразмерами hw=(x, y, z) вокруг центров (K, 3); 8 вершин и 12 треугольников на брус"""
    K = len(P)
    if K == 0: return np.zeros((0, 3), np.float32), np.zeros((0,), np.uint32)
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.asarray(hw, float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    V = (P[:, None, :] + c[None]).reshape(-1, 3).astype(np.float32); idx = (np.arange(K)[:, None, None] * 8 + np.array(f)[None]).reshape(-1).astype(np.uint32)
    return V, idx


LIVE_BATCH = int(os.environ.get('V8_LIVE_BATCH', 25)); LIVE_DELAY = float(os.environ.get('V8_LIVE_DELAY', 3.))               # P: агенты стартуют пачками по BATCH каждые DELAY с
def _box_edges(P, hw):
    """рёбра брусьев (как в _boxes): 12 рёбер на брус парами вершин → (K·24, 3)"""
    if len(P) == 0: return np.zeros((0, 3), np.float32)
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.asarray(hw, float)
    e = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return (P[:, None, None, :] + c[np.array(e)][None]).reshape(-1, 3).astype(np.float32)


RAIL_HW = tuple(float(v) for v in os.environ.get('V8_RAIL_HW', '.16,.11,.11').split(','))             # полуразмеры вагончика на рельсе (x вдоль рельса, y, z)


def _discs(P, R, res=16):
    """K центров (K, 3) → вершины (K·(res+1), 3) и индексы треугольников: плоские диски радиуса R в плоскости XZ (как Spore: Circle, rotation 90°)"""
    K = len(P)
    if K == 0: return np.zeros((0, 3), np.float32), np.zeros((0,), np.uint32)
    t = np.linspace(0, 2 * np.pi, res, endpoint=False); rim = np.stack([R * np.cos(t), np.zeros(res), R * np.sin(t)], 1)
    V = np.concatenate([P[:, None, :], P[:, None, :] + rim[None]], 1).reshape(-1, 3).astype(np.float32)
    tri = np.stack([np.zeros(res, np.int64), 1 + np.arange(res), 1 + (np.arange(res) + 1) % res], 1)
    idx = (np.arange(K)[:, None, None] * (res + 1) + tri[None]).reshape(-1).astype(np.uint32)
    return V, idx


WRAP_X = 0.                                                                                       # период по оси x (θ маятника = 2π); 0 — без шва. Задаёт main_pend.


def wrap_lines(V, colors=None):
    """отрезки-пары (2K, 3), пересекающие шов x = ±P/2, режем на две части в периодических копиях (а не тянем через поле); colors (2K, 4) режутся так же"""
    P = WRAP_X; V = np.asarray(V, float).reshape(-1, 3)
    if P <= 0 or len(V) < 2: return V, colors
    A, B = V[0::2], V[1::2]; far = np.abs(B[:, 0] - A[:, 0]) > P / 2
    if not far.any(): return V, colors
    ca = cb = None
    if colors is not None and np.ndim(colors) == 2 and len(colors) == len(V): ca, cb = np.asarray(colors)[0::2], np.asarray(colors)[1::2]
    keep = ~far; a, b = A[far], B[far]; sg = np.where(a[:, 0] < b[:, 0], 1., -1.)           # a левее b: короткий путь идёт влево от a через шов (-P/2), b — вправо от шва
    b2 = b.copy(); b2[:, 0] -= sg * P                                                          # b в копии, ближайшей к a
    seam = -sg * P / 2; t = (seam - a[:, 0]) / (b2[:, 0] - a[:, 0]); m = a + t[:, None] * (b2 - a); m1 = m.copy(); m1[:, 0] = seam; m2 = m.copy(); m2[:, 0] = seam + sg * P
    a2 = a.copy(); a2[:, 0] += sg * P                                                          # a в копии, ближайшей к b
    out = np.empty((2 * (int(keep.sum()) + 2 * int(far.sum())), 3)); n = 2 * int(keep.sum())
    out[0:n:2], out[1:n:2] = A[keep], B[keep]; k = int(far.sum()); out[n:n + 2 * k:2], out[n + 1:n + 2 * k:2] = a, m1; out[n + 2 * k::2], out[n + 2 * k + 1::2] = m2, b
    if ca is None: return out, colors
    cm = ca[far] + t[:, None] * (cb[far] - ca[far]); C = np.empty((len(out), colors.shape[1]))
    C[0:n:2], C[1:n:2] = ca[keep], cb[keep]; C[n:n + 2 * k:2], C[n + 1:n + 2 * k:2] = ca[far], cm; C[n + 2 * k::2], C[n + 2 * k + 1::2] = cm, cb[far]
    return out, C


SHOW_X = 0.                                                                                       # показывать периодические копии (±WRAP_X) на x ∈ [−SHOW_X, SHOW_X] (θ маятника: 8); 0 — только один период. Задаёт main_pend.


def clone_lines(V, colors=None):
    """отрезки-пары → копии со сдвигами k·WRAP_X, обрезанные по |x| ≤ SHOW_X (после wrap_lines: ни один отрезок не пересекает шов)"""
    P, X = WRAP_X, SHOW_X; V = np.asarray(V, float).reshape(-1, 3)
    if P <= 0 or X <= 0 or len(V) < 2: return V, colors
    pc = colors is not None and np.ndim(colors) == 2 and len(colors) == len(V); outV, outC = [], []
    for k in range(-int(np.ceil((X + P / 2) / P)), int(np.ceil((X + P / 2) / P)) + 1):
        W = V.copy(); W[:, 0] += k * P; A, B = W[0::2].copy(), W[1::2].copy(); ca = np.asarray(colors)[0::2].copy() if pc else None; cb = np.asarray(colors)[1::2].copy() if pc else None
        dx = B[:, 0] - A[:, 0]; t0 = np.zeros(len(A)); t1 = np.ones(len(A)); ok = np.ones(len(A), bool)
        for lim, sg in ((-X, 1.), (X, -1.)):                                              # x ≥ −X и x ≤ X
            num = sg * (A[:, 0] - lim); den = sg * dx                                  # inside: num + t·den ≥ 0
            with np.errstate(divide='ignore', invalid='ignore'):
                t = -num / den
            par = np.abs(den) < 1e-15; ok &= ~(par & (num < 0))
            ent = (~par) & (den > 0); lea = (~par) & (den < 0); t0 = np.where(ent, np.maximum(t0, t), t0); t1 = np.where(lea, np.minimum(t1, t), t1)
        ok &= t0 < t1
        if not ok.any(): continue
        a2 = A[ok] + t0[ok, None] * (B[ok] - A[ok]); b2 = A[ok] + t1[ok, None] * (B[ok] - A[ok]); n = len(a2); o = np.empty((2 * n, 3)); o[0::2], o[1::2] = a2, b2; outV.append(o)
        if pc:
            c2 = np.empty((2 * n, colors.shape[1])); c2[0::2] = ca[ok] + t0[ok, None] * (cb[ok] - ca[ok]); c2[1::2] = ca[ok] + t1[ok, None] * (cb[ok] - ca[ok]); outC.append(c2)
    if not outV: return np.zeros((0, 3)), (np.zeros((0, np.shape(colors)[1])) if pc else colors)
    return np.vstack(outV), (np.vstack(outC) if pc else colors)


def clone_tris(V, idx, colors=None):
    """треугольники → копии со сдвигами k·WRAP_X; оставляем те, что хоть одной вершиной попадают в |x| ≤ SHOW_X"""
    P, X = WRAP_X, SHOW_X; V = np.asarray(V, float).reshape(-1, 3); idx0 = np.asarray(idx); T = idx0.reshape(-1, 3)
    if P <= 0 or X <= 0 or len(T) == 0: return V, idx0, colors
    pc = colors is not None and np.ndim(colors) == 2 and len(colors) == len(V); outV, outI, outC = [], [], []; nv = 0; K = int(np.ceil((X + P / 2) / P))
    for k in range(-K, K + 1):
        x = V[T, 0] + k * P; keep = ((x >= -X) & (x <= X)).any(1) | ((x.min(1) <= -X) & (x.max(1) >= X))
        if not keep.any(): continue
        u, inv = np.unique(T[keep], return_inverse=True); W = V[u].copy(); W[:, 0] += k * P; outV.append(W); outI.append(inv.reshape(-1, 3) + nv); nv += len(u)
        if pc: outC.append(np.asarray(colors)[u])
    if not outV: return np.zeros((0, 3)), np.zeros(0, np.uint32), (np.zeros((0, np.shape(colors)[1])) if pc else colors)
    return np.vstack(outV), np.vstack(outI).ravel(), (np.vstack(outC) if pc else colors)


def wrap_tris(V, idx, colors=None):
    """треугольники с разбросом по x > P/2 — дублируем в периодические копии (вершины с x<0 сдвигаем на +P в одной копии, с x>0 на −P в другой), без растяжки через поле"""
    P = WRAP_X; V = np.asarray(V, float).reshape(-1, 3); idx0 = idx; idx = np.asarray(idx).reshape(-1, 3)
    if colors is not None and not (np.ndim(colors) == 2 and len(colors) == len(V)): colors = None if False else colors; pervert = False
    else: pervert = colors is not None
    if P <= 0 or len(idx) == 0: return V, idx0, colors
    x = V[idx, 0]; far = (x.max(1) - x.min(1)) > P / 2
    if not far.any(): return V, idx0, colors
    T = idx[far]; nv = len(V); newV, newC, newI = [V], ([np.asarray(colors)] if pervert else None), [idx[~far]]
    for sgn in ((1.,) if SHOW_X > 0 else (1., -1.)):                                       # при показе копий достаточно одной (остальные дадут clone_tris — без двойного наложения)
        u, inv = np.unique(T, return_inverse=True); W = V[u].copy(); W[(W[:, 0] < 0) if sgn > 0 else (W[:, 0] > 0), 0] += sgn * P
        newI.append(inv.reshape(-1, 3) + nv); newV.append(W); nv += len(u)
        if newC is not None: newC.append(np.asarray(colors)[u])
    return np.vstack(newV), np.vstack(newI).ravel(), (np.vstack(newC) if newC is not None else colors)


def _geom_node(V, kind, idx=None, col=(1., 1., 1., 1.)):
    """GeomNode напрямую из panda3d: kind='lines' — пары вершин (один GeomLines), 'tris' — индексированные треугольники (один GeomTriangles)"""
    from panda3d.core import GeomVertexData, GeomVertexFormat, Geom, GeomLines, GeomTriangles, GeomNode, GeomEnums
    from panda3d.core import GeomVertexArrayFormat, InternalName
    V = np.ascontiguousarray(V, np.float32).reshape(-1, 3); n = len(V)
    af = GeomVertexArrayFormat(); af.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point); af.add_column(InternalName.get_color(), 4, Geom.NT_float32, Geom.C_color)
    fmt = GeomVertexFormat(); fmt.add_array(af); fmt = GeomVertexFormat.register_format(fmt)
    D = np.empty((n, 7), np.float32); D[:, :3] = V; D[:, 3:] = col                         # вершинные цвета: без них шейдер Ursina красит такой узел в серый
    vd = GeomVertexData('gv', fmt, Geom.UH_dynamic); vd.unclean_set_num_rows(n); memoryview(vd.modify_array(0)).cast('B').cast('f')[:] = memoryview(D.ravel()).cast('B').cast('f')
    if kind == 'strips':
        from panda3d.core import GeomLinestrips
        prim = GeomLinestrips(Geom.UH_dynamic)
        for st, ln in idx: prim.add_consecutive_vertices(st, ln); prim.close_primitive()
    elif kind == 'lines': prim = GeomLines(Geom.UH_dynamic); prim.add_consecutive_vertices(0, n)
    else:
        prim = GeomTriangles(Geom.UH_dynamic); prim.set_index_type(GeomEnums.NT_uint32); ia = prim.modify_vertices(); ia.unclean_set_num_rows(len(idx)); memoryview(ia).cast('B').cast('I')[:] = memoryview(np.ascontiguousarray(idx, np.uint32)).cast('B').cast('I')
    if kind != 'strips': prim.close_primitive()
    g = Geom(vd); g.add_primitive(prim); node = GeomNode('gv'); node.add_geom(g); return node


def _make_batch():
    from ursina import Entity
    from panda3d.core import TransparencyAttrib

    class _Batch(Entity):
        """слой: один GeomNode; вершины в реальных координатах, a и b — масштаб/позиция самого узла (перестройка меша только при смене снимка)"""
        def __init__(self):
            super().__init__(); self._gn = None; self.col = (1., 1., 1., 1.); self.tk = 1.; self._thick = None; self.real_position = np.zeros(3); self.real_scale = np.ones(3)
        def set_geometry(self, V, idx=None, kind='lines', thick=None, colors=None, depth_write=True, on_top=False, fixed=None):
            """colors — (n, 4) цвета вершин (иначе один цвет слоя); depth_write=False — полупрозрачная поверхность не закрывает линии за ней"""
            if self._gn is not None: self._gn.remove_node(); self._gn = None
            if len(V) == 0: return
            if WRAP_X > 0 and kind == 'strips':                                                  # ломаные (строки/столбцы клеток) → пары вершин: резка по шву и копии работают только с парами
                Vs_ = np.asarray(V, float).reshape(-1, 3); pc_ = colors is not None and np.ndim(colors) == 2 and len(colors) == len(Vs_)
                ia = np.concatenate([np.arange(st, st + ln - 1) for st, ln in idx if ln > 1]) if len(idx) else np.zeros(0, int)
                if len(ia) == 0: return
                pr = np.empty(2 * len(ia), int); pr[0::2], pr[1::2] = ia, ia + 1; V = Vs_[pr]; colors = np.asarray(colors)[pr] if pc_ else colors; kind = 'lines'; idx = None
            if WRAP_X > 0:
                if kind == 'lines': V, colors = wrap_lines(V, colors)
                elif kind == 'tris' and idx is not None: V, idx, colors = wrap_tris(V, idx, colors)
                if SHOW_X > 0:
                    if kind == 'lines': V, colors = clone_lines(V, colors)
                    elif kind == 'tris' and idx is not None: V, idx, colors = clone_tris(V, idx, colors)
                    if len(V) == 0: return
            self._gn = self.attach_new_node(_geom_node(V, kind, idx, self.col if colors is None else colors)); self._gn.set_transparency(TransparencyAttrib.M_alpha)
            if not depth_write: self._gn.set_depth_write(False)
            if fixed is not None: self._gn.set_bin('fixed', fixed)                                        # порядок отрисовки: подложка (31) → линии скелета (32) → траектории в цель (60)
            if on_top: self._gn.set_depth_test(False); self._gn.set_bin('fixed', 60)                  # поверх всего (белые траектории, входящие в цель)
            if kind != 'tris': self._thick = thick or 2; self._gn.set_render_mode_thickness(max(1., self._thick * self.tk))
            else: self._gn.set_two_sided(True)
            self._gn.set_depth_offset(1)                                   # поверх пола (y = 0), без z-fighting
        def set_tk(self, k):
            """множитель толщины (1 + колесо: размер траекторий); применяется к уже построенной геометрии"""
            self.tk = k
            if self._gn is not None and self._thick is not None: self._gn.set_render_mode_thickness(max(1., self._thick * k))
        def apply_transform(self, a, b, **kw): self.position = b; self.scale = a
    return _Batch


def _Batch():
    global _Batch
    _Batch = _make_batch(); return _Batch()


class GrowView:
    """Ursina-часть: создаёт по Scalable на слой геометрии и перерисовывает при смене снимка. Импорт Ursina — лениво."""
    # цвета: центральная линия + зерно — зелёные; боковые линии — оранжевые (гало бледнее ядра); зигзаг строк и их узлы — светло-серые (контраст к зелёному и оранжевому на тёмно-синем); текущая клетка ярче, готовые приглушены
    COL = {'points_done': (.62, .62, .68, .9), 'segs_rows_done': (.74, .74, .80, .85), 'segs_center_done': (.35, .78, .45, .9), 'segs_core_done': (.88, .60, .30, .9), 'segs_halo_done': (.72, .55, .38, .7),
           'points_cur': (.92, .92, .96, 1), 'segs_rows_cur': (.95, .95, 1.0, 1), 'segs_center_cur': (.20, 1.0, .30, 1), 'segs_core_cur': (1.0, .55, .10, 1), 'segs_halo_cur': (1.0, .70, .40, .6),
           'points_seed': (.2, 1.0, .3, 1), 'points_queue': (.35, .9, .85, .8), 'segs_goal': (.55, 1.0, .65, .9),
           'segs_field': (.50, .75, 1.0, .6), 'segs_growlim': (.50, .75, 1.0, .25), 'segs_ell': (1.0, .85, .25, 1.0), 'segs_ellp': (1.0, .85, .25, .35)}   # граница поля (посев) и граница роста (XLV+GM): бледно-голубые

    def __init__(s, zoom_manager, ax=(0, 1), halo=HALO, spore_manager=None):
        s.sm = spore_manager; s._sz0 = float(spore_manager.size) if spore_manager is not None else 1.; s._pts = {}; s._built_size = None; s.vis = [True] * 8; s.skel = 'normal'; s.zm = zoom_manager; s.ax = ax; s.halo = halo; s.layers = {}; s.last = None; s.times = TimeTable(); s.snap = None; s.vmode = True; s.cmode = False; s.tmode = True; s._auto = True; s.tview = 'both'; s._traj = None; s._tjob = None; s._tstat = ''; s._tfresh = False; s._vcache = {}; s._tcache = {}; s._mcache = {}; s._gkey = {}; s._vfresh = False; s.dinfo = {}; s.runlog = None; s.vcalc = 'q'; s._stj = None; s._tfail = None; s.pics_dirs = []; s._pics_note = ''; s._prog = ''; s._stinfo = ''; s._vsig = None; s._vres = None; s.vinfo = ''; s._tpart = {}; s._tprog = ''; s._live = None; s._live_on = False; s._live_t = 0.; s._live_used = set(); s._live_tab = None; s._live_rail = None

    TINT = ('points_done', 'points_cur', 'segs_rows_done', 'segs_center_done', 'segs_core_done', 'segs_halo_done', 'segs_rows_cur', 'segs_center_cur', 'segs_core_cur', 'segs_halo_cur')

    @staticmethod
    def tinted(col, k, past=False):
        """оттенок слоя k: 0 — как есть; 1 — холодный (R↔B); 2 — жёлто-зелёный (R→G→R); прошлые слои приглушены"""
        r, g_, b, a = col
        r, g_, b = ((r, g_, b), (b, g_, r), (g_, r, b))[k % 3]
        return (r * .85, g_ * .85, b * .85, a * .9) if past else (r, g_, b, a)

    def toggle_layer(s, k):
        s.vis[k] = not s.vis[k]
        if s.snap is not None: s.draw(s.snap)

    def toggle_skeleton(s):
        s.skel = 'time' if s.skel == 'normal' else 'normal'
        if s.snap is not None: s.draw(s.snap)

    def skeleton_label(s):
        has = any(c.get('Gt') is not None for c in ((s.snap or {}).get('cells') or [])) or ((s.snap or {}).get('cell') or {}).get('Gt') is not None
        return 'skeleton: %s%s' % (s.skel, ' (no Gt yet)' if s.skel == 'time' and not has else '')

    def _layer(s, name, col=None):
        """один объект Ursina на слой (один GeomNode, один примитив); зум/сдвиг — трансформ узла, вершины в реальных координатах"""
        if name not in s.layers:
            e = _Batch(); e.col = tuple(float(x) for x in (col or s.COL[name.split('@')[0]]))
            s.zm.register_object(e, name='growview_' + name); s.layers[name] = e
            if name.split('@')[0] in s.TRAJ_THICK: e.tk = s.tk()
        elif col is not None: s.layers[name].col = tuple(float(x) for x in col)
        return s.layers[name]

    TRAJ_THICK = ('lv_ff', 'lv_sf', 'lv_pts_edge', 'traj_floor', 'traj_surf')      # линии траекторий масштабируются клавишей 1 + колесо (как споры)

    def tk(s):
        """множитель размера относительно начального SporeManager.size"""
        return (s.sm.size / s._sz0) if (s.sm is not None and s._sz0) else 1.

    def _set_layer(s, name, arr, col=None):
        e = s._layer(name, col); arr = np.asarray(arr, float).reshape(-1, 3)
        if name.startswith('points'):
            s._pts[name] = arr; R = .5 * SPORE_SCALE * (s.sm.size if s.sm is not None else 1.); s._built_size = None if s.sm is None else s.sm.size    # диски как у Spore (Circle r = .5·size), в мире
            e.set_geometry(*_discs(arr, R), kind='tris')
        else: e.set_geometry(arr, kind='lines', thick=2)

    def _set_strips(s, name, strips, col=None):
        e = s._layer(name, col); strips = [np.asarray(x, float).reshape(-1, 3) for x in strips if len(x) > 1]
        if not strips: e.set_geometry(np.zeros((0, 3))); return
        starts = np.cumsum([0] + [len(x) for x in strips[:-1]]); e.set_geometry(np.concatenate(strips, 0), [(int(a), len(x)) for a, x in zip(starts, strips)], kind='strips', thick=2)

    def refresh(s):
        """размер спор ('1') поменялся — пересобрать диски (зум этого не требует)"""
        if s.sm is not None and getattr(s, '_built_size', None) not in (None, s.sm.size):
            for name, arr in list(s._pts.items()): s._set_layer(name, arr)
        if s.sm is not None and getattr(s, '_tr_size', s._sz0) != s.sm.size:                                       # траектории: толщина линий, диски и маркеры — тем же размером
            s._tr_size = s.sm.size; k = s.tk()
            for nm, e in s.layers.items():
                if nm.split('@')[0] in s.TRAJ_THICK: e.set_tk(k)
            s._gkey.pop('traj', None)
            if s.tmode and not s._live_on: s._draw_traj()

    def draw(s, snap):
        """нарисовать снимок (None — очистить). Время отрисовки уходит в TimeTable по имени паузы."""
        t0 = time.perf_counter(); s.snap = snap
        if snap is None: s.draw_hist = {}
        elif id(snap) in getattr(s, 'draw_hist', {}):                  # a past snapshot (Z): draw it again, restore the draw-time table as it was
            s.times.draw, s.times.n = (dict(x) for x in s.draw_hist[id(snap)]); t0 = None
        z3 = np.zeros((0, 3)); snap_ = snap or {}
        cur_k = snap_.get('layer_k', 0); nl = snap_.get('layers_n', 1); cbl = snap_.get('cells_by_layer') or {}; layers = snap_.get('layers') or ()
        s._nk = max(getattr(s, '_nk', 0), nl)
        for k in range(s._nk):                                         # по слою k: текущий — всё; прошлые — готовые клетки приглушённо; будущие/скрытые — пусто
            if snap is None or k > cur_k or not s.vis[k]: g, st, past = {}, {}, False
            elif k == cur_k: g = build_geometry(snap, s.ax, s.halo, s.skel); st = build_strips(snap, s.ax, s.skel); past = False
            else:
                cl = cbl.get(layers[k]) if k < len(layers) else None; past = True
                g = build_geometry({'cells': cl or []}, s.ax, s.halo, s.skel); st = build_strips({}, s.ax, s.skel, cells=cl or [], with_cur=False)
            for grp in s.TINT:
                col = s.tinted(s.COL[grp], k, past); name = '%s@%d' % (grp, k)
                if grp.startswith('points'): s._set_layer(name, g.get(grp, z3), col)
                else: s._set_strips(name, st.get(grp, []), col)
        gp = build_geometry(snap, s.ax, s.halo, s.skel) if snap is not None else {}
        for name in ('points_seed', 'points_queue'): s._set_layer(name, gp.get(name, z3))
        s._draw_vsurf(snap); s.dinfo.update(mesh_hit=getattr(s, '_mh', 0), mesh_built=getattr(s, '_mm', 0))
        if s.runlog is not None and snap is not None:
            try: s.runlog.atlas(snap)
            except Exception as e: print('[runlog] atlas: %r' % (e,))
        gs = s._goal(); fixed = {'segs_goal': [np.concatenate([gs[:, 0], gs[-1:, 1]], 0)], 'segs_field': [s._box(0.)], 'segs_growlim': [s._box(getattr(__import__('src.algo.growN', fromlist=['GMV']), 'GMV', 0.))]}   # цель, граница поля и роста
        try:                                                                                                  # №14: границы эллипсов посева (SEEDELL): текущая — ярко, прежние ступени — тоньше
            G_ = __import__('src.algo.growN', fromlist=['ELLS']); Ss = list(G_.ELLS.get(snap.get('u') if snap is not None else None, [])) if getattr(G_, 'SEEDELL', 0) else []
            def _el(S): return [proj(np.stack([l[:, 0] if i == s.ax[0] else l[:, 1] if i == s.ax[1] else np.zeros(len(l)) for i in range(G_.N)], 1), s.ax) for l in G_.ell_lines(S)]
            key = tuple(round(x, 3) for x in Ss)
            if key != getattr(s, '_ell_key', None): s._ell_cache = ([x for S in Ss[:-1] for x in _el(S)], _el(Ss[-1]) if Ss else []); s._ell_key = key
            fixed['segs_ellp'], fixed['segs_ell'] = s._ell_cache
        except Exception as e: print('[ell] %r' % (e,), flush=True)
        for name, strips in fixed.items(): s._set_strips(name, strips)
        if snap is not None and t0 is not None:
            s.times.add_draw(snap.get('phase', '?'), time.perf_counter() - t0); s.draw_hist = getattr(s, 'draw_hist', {}); s.draw_hist[id(snap)] = (dict(s.times.draw), dict(s.times.n))

    # ---- поверхность V (src/stepper/vsurf.py): V — по слоям (7/8/9), B — общий слой V* (победители по слоям); один меш на слой и тип ----
    def toggle_vsurf(s):
        s._auto = False; s.vmode = not s.vmode; print('[vsurf] per-layer surfaces %s' % ('on' if s.vmode else 'off'), flush=True)
        if s.snap is not None: s.draw(s.snap)

    def toggle_common(s):
        s._auto = False; s.cmode = not s.cmode; print('[vsurf] common V* layer %s' % ('on' if s.cmode else 'off'), flush=True)
        if s.snap is not None: s.draw(s.snap)

    VCALC = (('v', 'V old'), ('q', 'V by Q (lin)'), ('qquad', 'V by Q (quad)'))

    def cycle_vcalc(s):
        """расчёт V для поверхностей V/B и траекторий: старый → по Q (lin) → по Q (quad); solve8.Graph.set_calc (v_kink_q.md)"""
        ks = [k for k, _ in s.VCALC if k != 'qquad' or s._cellM() == 3]; s.vcalc = ks[(ks.index(s.vcalc) + 1) % len(ks)] if s.vcalc in ks else ks[0];      # Q quad только при M=3
        print('[vsurf] V calc: %s' % dict(s.VCALC)[s.vcalc], flush=True)
        if s.snap is not None: s.draw(s.snap)

    TVIEWS = ('both', 'floor', 'surface')

    def toggle_traj(s):
        """T: траектории выкл → пол+поверхность → только пол → только поверхность → выкл (слово пользователя 2026-10-12)"""
        s._auto = False
        if not s.tmode: s.tmode = True; s.tview = 'both'
        elif s.tview == 'surface': s.tmode = False
        else: s.tview = s.TVIEWS[s.TVIEWS.index(s.tview) + 1]
        s._tfresh = False; print('[vsurf] final trajectories: %s' % (s.tview if s.tmode else 'off'), flush=True)
        if s.snap is not None: s.draw(s.snap)

    def _tlabel(s):
        """подпись траекторий: computing… / computed (только что) / cached (из кэша: T только переключает видимость)"""
        if s._tprog: return 'computing ' + s._tprog + ' (shown as they arrive)'
        if s._traj is None: return s._tstat or 'computing...'
        return ('computed: ' if s._tfresh else 'cached: ') + s._tstat

    def vlabel(s):
        on = [n for n, f in (('V per layer', s.vmode), ('V* common', s.cmode), ('trajectories on %s (%s)' % (s.tview, s._tlabel()), s.tmode)) if f]
        return ('%s [%s, V %s]: %s' % (' + '.join(on), dict(s.VCALC)[s.vcalc], 'computed' if s._vfresh else 'cached', s.vinfo or 'on')) if on else ''

    def _vcells(s, snap):
        """{u: [cell_dict]} — готовые клетки показанных слоёв (до текущего включительно); скрытые 7/8/9 слои в расчёте участвуют (E3 — переходы между слоями)"""
        layers = snap.get('layers') or (); cur = snap.get('layer_k', 0); cbl = snap.get('cells_by_layer') or {}; out = {}
        for k, u in enumerate(layers):
            if k > cur: break
            cl = (snap.get('cells') if k == cur else cbl.get(u)) or []
            if cl: out[u] = cl
        if os.environ.get('V8_ONLY_GOAL') == '1' and len(out) < len(layers): return {}                  # режим первой споры: V/траектории/карта ошибки — только когда выросли ОБА слоя (+1 и −1), а не после первого шага слоя
        return out

    VNAMES = ('surf', 'pts', 'inf', 'long', 'cross', 'cen', 'goal', 'gcen', 'under', 'undc')

    def _draw_vsurf(s, snap):
        from . import vsurf as VS
        s._stinfo = s._stage_text(snap)
        z3 = np.zeros((0, 3)); on = snap is not None and (s.vmode or s.cmode or s.tmode) and not (s._auto and snap.get('phase') != 'done')      # V и траектории включены по умолчанию (слово пользователя 2026-10-12), но показываются/считаются, только когда рост закончен (phase done: в режиме N спор — оба слоя); любая клавиша V/B/T передаёт управление пользователю
        def clear(pre, ks):
            for k in ks:
                s._gkey.pop(pre + '@%d' % k, None)
                for n in s.VNAMES:
                    if pre + n + '@%d' % k in s.layers: s.layers[pre + n + '@%d' % k].set_geometry(z3)
        ks = range(max(getattr(s, '_nk', 0), 3)); s._vfresh = False; s._mh = s._mm = 0; s.dinfo = {}
        if not on: clear('v', ks); clear('c', ks); s._clear_traj(); return                                        # кэши V и траекторий остаются: повторное включение того же атласа/режима — без пересчёта
        cl = s._vcells(snap)
        if not cl and os.environ.get('V8_ONLY_GOAL') == '1' and snap is not None and (snap.get('layers') or ()):
            s.vinfo = 'waiting: first-spore mode needs BOTH layers grown (N / LMB until layer -1 is finished)'; s._vres = None; clear('v', ks); clear('c', ks); s._clear_traj(); return
        sig = (s.vcalc,) + tuple((u, len(c), id(c[0]['G']), id(c[-1]['G'])) for u, c in cl.items())
        if sig != s._vsig or s._vres is None:
            s._vsig = sig; s._tprog = ''; hit = s._tcache.get(sig); s._traj = None if hit is None else hit[:2]; s._tstat = '' if hit is None else hit[2]; s._tfresh = False
            if sig in s._vcache:
                s._vres, s.vinfo = s._vcache[sig]; s.dinfo['vcalc'] = 'hit'
                if s.runlog is not None: s.runlog.vcalc(s.vcalc, s._vres[3], sum(len(c) for c in cl.values()), 0., cache='hit')
            else:
                s._vres = None
                try:
                    t0 = time.perf_counter(); Vs, info = VS.solve_layers(cl, calc=s.vcalc); s._vres = (Vs, VS.vscale(Vs), cl, info)
                    s.vinfo = s._vstat(info, sum(len(c) for c in cl.values()))
                    print('[vsurf] ' + s.vinfo + ' (total %.0f ms)' % (1e3 * (time.perf_counter() - t0)), flush=True)
                    if s.runlog is not None: s.runlog.vcalc(s.vcalc, info, sum(len(c) for c in cl.values()), time.perf_counter() - t0)
                    if len(s._vcache) >= 8: s._tcache.pop(next(iter(s._vcache)), None); s._vcache.pop(next(iter(s._vcache)))                                  # кэш по (атлас, режим): возврат к прежнему режиму — из кэша
                    s._vcache[sig] = (s._vres, s.vinfo); s._vfresh = True; s.dinfo['vcalc'] = 'miss'
                except Exception as e: s.vinfo = 'solve failed: %r' % (e,); print('[vsurf] ' + s.vinfo, flush=True); s.vmode = s.cmode = s.tmode = False
        if s._vres is None: clear('v', ks); clear('c', ks); s._clear_traj(); return
        s.dinfo.setdefault('vcalc', 'same')
        s._draw_traj()
        Vs, vmax, cl, info = s._vres; layers = snap.get('layers') or (); R = .5 * SPORE_SCALE * (s.sm.size if s.sm is not None else 1.)
        for pre, flag in (('v', s.vmode), ('c', s.cmode)):                                                        # v — отдельные слои (видимость 7/8/9), c — общий слой V*: победители, видимость слоя тоже по 7/8/9
            for k, u in enumerate(layers):
                if not flag or u not in cl or not s.vis[k]: clear(pre, [k]); continue
                gk = pre + '@%d' % k; mkey = (s._vsig, u, pre, s.skel, tuple(s.ax), round(R, 5))
                if s._gkey.get(gk) == mkey: s._mh += 1; continue                                                   # слой уже показан с тем же содержимым — геометрию не трогаем
                m = s._mcache.get(mkey[:5])
                if m is None:
                    m = VS.build_mesh(cl[u], Vs[u], s.ax, vmax, win=info['wins'][u] if pre == 'c' else None, goalcols=info['goalcols'][u], skel=s.skel)
                    if len(s._mcache) >= 24: s._mcache.pop(next(iter(s._mcache)))
                    s._mcache[mkey[:5]] = m
                s._mm += 1; s._gkey[gk] = mkey; w = (1., 1., 1., 1.); nm = lambda t: pre + t + '@%d' % k
                lc = VS.line_color(s.tinted(s.COL['segs_core_done'], k)[:3])                                       # скелет — оттенок группы слоя, осветлённый (читается на градиенте без чёрной подложки)
                s._layer(nm('surf'), w).set_geometry(m['verts'], m['tris'], kind='tris', colors=m['colors'], depth_write=False)
                Dp, ip = _discs(m['pts'], R); s._layer(nm('pts'), w).set_geometry(Dp, ip, kind='tris', colors=np.repeat(m['pcol'], 17, axis=0))
                for nmu, arr, th in ((('under', m['long'], VS.LONG_W), ('undc', m['center'], VS.CENTER_W)) if VS.UNDER_A > 0 else ()):                                  # тёмная подложка под продольными линиями (синие читаются на синей части градиента)
                    s._layer(nm(nmu), w).set_geometry(arr, kind='lines', thick=th + VS.UNDER_DW, colors=np.tile((.04, .04, .10, VS.UNDER_A), (len(arr), 1)), depth_write=False, fixed=31)
                for t, a_, th, col, top in (('long', VS.LONG_A, VS.LONG_W, lc, False), ('cross', VS.CROSS_A, VS.CROSS_W, lc, False), ('cen', VS.CENTER_A, VS.CENTER_W, lc, False), ('goal', VS.GOAL_A, VS.GOAL_SIDE_W, (1., 1., 1.), True), ('gcen', VS.GOAL_A, VS.GOAL_W, (1., 1., 1.), True)):
                    arr = m[{'cen': 'center', 'gcen': 'goalc'}.get(t, t)]; s._layer(nm(t), w).set_geometry(arr, kind='lines', thick=th, colors=np.tile((*col, a_), (len(arr), 1)), depth_write=False, on_top=top, fixed=None if top else 32)       # goal/gcen — белые, поверх всего (центральный столбец жирнее боковых)
                Di, ii = _discs(m['inf'], R); s._layer(nm('inf'), w).set_geometry(Di, ii, kind='tris', colors=np.tile((.55, .55, .58, .9), (len(Di), 1)))

    def _clear_traj(s):
        s._gkey.pop('traj', None)
        for n in ('traj_floor', 'traj_surf', 'traj_pf', 'traj_ps', 'traj_mk'):
            if n in s.layers: s.layers[n].set_geometry(np.zeros((0, 3)))

    # ---- живое проигрывание траекторий (P) и агент от точки взгляда (O): src/stepper/liveplay.py; в кадре дорисовываются только новые сегменты ----
    def _live_G(s):
        return None if s._vres is None else s._vres[3].get('G')

    def _live_table(s):
        """таблица Q (b2: Graph.build_table) в фоновом потоке при первом использовании: до готовности агент считает точным поиском (~50 мс), потом lookup (~8 мс)"""
        G = s._live_G()
        if G is None or getattr(G, 'tab', 1) is not None or (s._live_tab is not None and s._live_tab.is_alive()): return
        s._live_tab = threading.Thread(target=G.build_table, daemon=True); s._live_tab.start()

    def _live_clear(s):
        z3 = np.zeros((0, 3))
        for nm in list(s._live_used): s._layer(nm, (1., 1., 1., 1.)).set_geometry(z3)
        s._live_used = set(); s._live_rail = None

    def live_toggle(s):
        """P: проиграть траектории (300 агентов) как анимацию; повторно — выключить и вернуть статичные"""
        if s._live_on:
            s._live_on = False; s._live_clear(); s._gkey.pop('traj', None); print('[live] off', flush=True)
            if s.snap is not None: s.draw(s.snap)
            return
        if s._traj is None: print('[live] no trajectories yet (wait for the calculation)', flush=True); return
        from .liveplay import LivePlay
        s._live = LivePlay(s.ax); R0 = s._traj[0]; s._live_q = [{k: R0[k][i:i + LIVE_BATCH] for k in ('paths', 'V', 'U')} for i in range(0, len(R0['paths']), LIVE_BATCH)]; s._live.add(s._live_q.pop(0)); s._live_next = time.perf_counter() + LIVE_DELAY; s._live_on = True; s._live_t = time.perf_counter(); s._clear_traj(); s._live_table(); print('[live] playing %d trajectories: batches of %d every %.1f s' % (len(s._traj[0]['paths']), LIVE_BATCH, LIVE_DELAY), flush=True)

    def live_launch(s, xv):
        """O: агент от точки взгляда xv=(x, v) — новый старт сразу (таблица Q / точный поиск), путь раскрывается вместе с остальными"""
        G = s._live_G()
        if G is None or s._vres is None: print('[live] no V yet', flush=True); return
        from .liveplay import LivePlay
        from . import vsurf as VS
        s._live_table(); t0 = time.perf_counter(); res = VS.agent_paths(G, cl=s._vres[2], starts=np.array([[float(xv[0]), float(xv[1])]]))
        if not s._live_on:
            s._live = LivePlay(s.ax); s._live_on = True; s._live_t = time.perf_counter(); s._clear_traj()
        s._live.add(res); print('[live] agent from (%.2f, %.2f): %s, %d points, %.0f ms' % (xv[0], xv[1], res['status'][0], len(res['paths'][0]), 1e3 * (time.perf_counter() - t0)), flush=True)

    def live_update(s):
        """каждый кадр (main.update): шаг проигрывания → новые сегменты в хвостовой слот пула слоёв + головы агентов"""
        if not s._live_on or s._live is None: return
        now = time.perf_counter()
        if getattr(s, '_live_q', None) and now >= s._live_next: s._live.add(s._live_q.pop(0)); s._live_next = now + LIVE_DELAY            # следующая пачка стартует по таймеру
        dt = min(now - s._live_t, .1); s._live_t = now; o = s._live.step(dt); w = (1., 1., 1., 1.); k = o['slot']; t = o['tail']
        from . import vsurf as VS
        if o['new'] and t['floor'] is not None:
            for nm, arr, col, th in (('lv_ff@%d' % k, t['floor'], t['fcol'], VS.TRAJ_FW), ('lv_sf@%d' % k, t['surf'], t['scol'], VS.TRAJ_SW)):
                s._layer(nm, w).set_geometry(arr, kind='lines', thick=th, colors=col, depth_write=False, on_top=True); s._live_used.add(nm)
        fl = s.tmode and s.tview in ('both', 'floor'); sf = s.tmode and s.tview in ('both', 'surface')                 # P учитывает режим T: пол / поверхность / оба / выкл (слой-слот просто включается/выключается)
        for nm in s._live_used:
            if nm.startswith('lv_ff@') or nm == 'lv_head': s._layer(nm, w).enabled = fl
            elif nm.startswith('lv_sf@'): s._layer(nm, w).enabled = sf
        H, HC = o['heads']; R = .5 * SPORE_SCALE * (s.sm.size if s.sm is not None else 1.) * VS.TRAJ_PT * 1.6
        if s._live_rail is None:                                                                         # «рельс»: реальная 1D-система (x по оси x, v — скорость точки) в мировой точке (0, 0, RAIL_Z); цель |x| ≤ ρ — отрезок; масштаб x как на плоскости
            xl = float(os.environ.get('XL', 3)) + float(os.environ.get('GMX', 3)); rho = float(os.environ.get('RHO', .2)); z = VS.RAIL_Z
            s._layer('lv_rail', w).set_geometry(np.array([[-xl, 0, z], [xl, 0, z]], float), kind='lines', thick=3, colors=np.tile((.75, .75, .8, .9), (2, 1)), depth_write=False); s._live_used.add('lv_rail')
            s._layer('lv_goal', w).set_geometry(np.array([[-rho, 0, z], [rho, 0, z]], float), kind='lines', thick=8, colors=np.tile((1., 1., 1., 1.), (2, 1)), depth_write=False, on_top=True); s._live_used.add('lv_goal'); s._live_rail = True
        RP = H.copy(); RP[:, 2] = VS.RAIL_Z; hw_ = tuple(h_ * s.tk() for h_ in RAIL_HW); D2, i2 = _boxes(RP, hw_); s._layer('lv_pts', w).set_geometry(D2, i2, kind='tris', colors=np.repeat(HC, 8, axis=0) if len(HC) else None, depth_write=False, on_top=True); s._live_used.add('lv_pts')
        E = _box_edges(RP, hw_); s._layer('lv_pts_edge', w).set_geometry(E, kind='lines', thick=2, colors=np.tile((0., 0., 0., 1.), (len(E), 1)) if len(E) else None, depth_write=False, on_top=True); s._live_used.add('lv_pts_edge')       # чёрная рамка у каждой вагонетки       # по точке на агента: x(t), цвет — текущее управление
        D, ii = _discs(H, R); s._layer('lv_head', w).set_geometry(D, ii, kind='tris', colors=np.repeat(HC, 17, axis=0) if len(HC) else None, depth_write=False, on_top=True); s._live_used.add('lv_head')

    def _draw_traj(s):
        """итоговые траектории (клавиша T): TRAJ_N стартов, агент solve8 (look) на графе показанного атласа считается в ФОНОВОМ потоке (UI не стоит; подпись «computing…»); готовый путь — на полу и на поверхности V*, линии и точки в цвет управления (u>0 фиолетовый, u<0 красный, u=0 серый), поверх всего"""
        from . import vsurf as VS
        if not s.tmode or s._live_on: s._clear_traj(); return                                                  # идёт живое проигрывание (P) — статичные траектории скрыты
        if s._traj is None:
            s._clear_traj()
            if s._tfail == s._vsig: s._tstat = 'failed (see console)'; return
            if not any(len(c) for c in s._vres[2].values()): s._tstat = 'no cells'; return
            if s._stj is None or not s._stj['th'].is_alive():
                s._tstat = 'computing...'; job = dict(items=[dict(kind='T', calc=s._vsig[0], sig=s._vsig, cl=s._vres[2])], q=[], err=None); job['th'] = threading.Thread(target=s._stage_run, args=(job,), daemon=True); s._stj = job; job['th'].start()
            return
        res, m = s._traj; w = (1., 1., 1., 1.); R = .5 * SPORE_SCALE * (s.sm.size if s.sm is not None else 1.) * VS.TRAJ_PT; tkey = (id(m), s.tview, round(R, 5))
        s.dinfo['traj_data'] = 'cached' if not s._tfresh else 'computed'
        if s._gkey.get('traj') == tkey: return
        s._gkey['traj'] = tkey; z3, z4 = np.zeros((0, 3)), np.zeros((0, 4)); fl, sf = s.tview in ('both', 'floor'), s.tview in ('both', 'surface')
        for nm, arr, col, th, on in (('traj_floor', m['floor'], m['fcol'], VS.TRAJ_FW, fl), ('traj_surf', m['surf'], m['scol'], VS.TRAJ_SW, sf)):
            s._layer(nm, w).set_geometry(arr if on else z3, kind='lines', thick=th, colors=col if on else z4, depth_write=False, on_top=True)
        for nm, pts, col, on in (('traj_pf', m['pf'], m['pfc'], fl), ('traj_ps', m['ps'], m['psc'], sf)):
            D, ii = _discs(pts if on else z3, R); s._layer(nm, w).set_geometry(D, ii, kind='tris', colors=np.repeat(col if on else z4, 17, axis=0), depth_write=False, on_top=True)
        mk = m['mk'] if (fl or sf) else z3; D, ii = _discs(mk, R * 2.4); s._layer('traj_mk', w).set_geometry(D, ii, kind='tris', colors=np.tile((1., 1., 1., 1.), (len(D), 1)), depth_write=False, on_top=True)       # белые маркеры на границе области: траектория вышла за выращенную область

    def _sig(s, calc, cl): return (calc,) + tuple((u, len(c), id(c[0]['G']), id(c[-1]['G'])) for u, c in cl.items())

    def _traj_stat(s, res, secs):
        t = s._traj_stat0(res, secs)
        if os.environ.get('SYS') == 'pend' and os.environ.get('GOALSHAPE', 'box') == 'ball': t += '  [T_ref pend — для ящика ±.1; для круга чуть занижен]'
        return t

    def _traj_stat0(s, res, secs):
        st = np.asarray(res['status'], object); fl = np.flatnonzero(st == 'failed'); ok = st == 'reached'; nout = int((st == 'out_of_area').sum()); nin = len(st) - nout; w = (st != 'out_of_area') & ~res['left_work']
        return '%d/%d reached (work zone %d/%d), %d out of area, T median %.2f, switches %.1f, %.1f s' % (int(ok.sum()), nin, int((ok & ~res['left_work']).sum()), int(w.sum()), nout, float(np.median(res['T'][ok])) if ok.any() else float('nan'), float(res['sw'][st != 'out_of_area'].mean()) if nin else 0., secs) + ('' if not len(fl) else '; NOT REACHED: ' + '; '.join('#%d start (%.2f,%.2f) end (%.3f,%.3f) last u=%+g, V*(end) %s' % (i, *res['X0'][i][:2], *res['paths'][i][-1][:2], res['U'][i][-1], ('%.3f' % res['V'][i][-1]) if np.isfinite(res['V'][i][-1]) and res['V'][i][-1] < 1e8 else 'inf') for i in fl[:8]))

    def _vstat(s, info, ncells): return '%d cells in %d layers, %d nodes (V finite %.0f%%), V max %.2f, solve8 %.2f s / %d passes%s' % (ncells, len(info.get('layers') or {}), info['n'], 100 * info.get('finite', 0.), info.get('vmax', 0.), info['t'], info['passes'], ('' if info.get('converged', True) else ' (NOT CONVERGED, capped; known defect of the old V)') + s._qerr_text(info))

    @staticmethod
    def _qerr_text(info):
        q = info.get('qerr') or {}
        t = ('; Q-exact entry |err| at nodes ' + ', '.join('u=%+g max %.3f med %.3f bias %+.3f (%d nodes)' % (u, d['max'], d['med'], d['bias'], d['n']) for u, d in q.items() if d['n'])) if q else ''
        return t

    def _put_v(s, sig, entry):
        if len(s._vcache) >= 8: s._tcache.pop(next(iter(s._vcache)), None); s._vcache.pop(next(iter(s._vcache)))
        s._vcache[sig] = entry

    def _put_t(s, sig, entry):
        if len(s._tcache) >= 8: s._tcache.pop(next(iter(s._tcache)))
        s._tcache[sig] = entry

    # ---- этапы после роста (слово пользователя 2026-10-12 ~16:35): расчёты V всех вариантов и траектории каждого; ЛКМ — следующий этап (пауза между ними, видно промежуточное), ПКМ — все оставшиеся подряд ----
    def _cellM(s):
        cl = s._vcells(s.snap) if s.snap is not None else {}
        return int(np.shape(next(iter(cl.values()))[0]['G'])[1]) if cl else 0

    def stage_todo(s, snap):
        """список ещё не посчитанных этапов (dict kind 'V'|'T', calc, sig, cl) — только когда рост закончен (phase done); готовое лежит в кэшах _vcache/_tcache; Q quad — только при M=3"""
        if snap is None or snap.get('phase') != 'done': return []
        cl = s._vcells(snap)
        if not cl: return []
        M = int(np.shape(next(iter(cl.values()))[0]['G'])[1]); out = []
        for calc, _ in s.VCALC:
            if calc != s.vcalc: continue                                                                 # только выбранный расчёт (Y переключает; его V и траектории считаются сами при показе): ПКМ не тянет чужие долгие этапы (старый V не сходится), траектории приходят сразу после V без клавиши T
            if calc == 'qquad' and M != 3: continue
            sig = s._sig(calc, cl)
            if sig not in s._vcache: out.append(dict(kind='V', calc=calc, sig=sig, cl=cl))
            if sig not in s._tcache: out.append(dict(kind='T', calc=calc, sig=sig, cl=cl))
        return out

    def stage_next(s, snap, all_=False):
        """запустить следующий этап (all_ — все оставшиеся подряд) в фоновом потоке; → True, если что-то было запущено/идёт"""
        if s._stj is not None and s._stj['th'].is_alive(): print('[calc] a stage is still running: %s' % s._prog, flush=True); return True
        todo = s.stage_todo(snap)
        if not todo: print('[calc] all stages done', flush=True); s._stinfo = ''; return False
        items = todo if all_ else todo[:1]; job = dict(items=items, q=[], err=None, left=len(todo)); job['th'] = threading.Thread(target=s._stage_run, args=(job,), daemon=True); s._stj = job; job['th'].start(); return True

    def _stage_run(s, job):
        """поток-читатель: запускает calc_worker.py (отдельный процесс — счёт вне GIL окна), печатает его прогресс как «[calc] k/N <имя> …», складывает результаты в job['q'] (poll → кэши)"""
        import subprocess, pickle
        from . import vsurf as VS
        names = dict(s.VCALC); items = job['items']; n = len(items); tag = lambda i: '[calc] %d/%d %s %s' % (i + 1, n, names[items[i]['calc']], 'V' if items[i]['kind'] == 'V' else 'trajectories')
        try:
            p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calc_worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=os.environ.copy()); job['proc'] = p
            pickle.dump(dict(items=[(it['kind'], it['calc'], it['cl']) for it in items], pics=list(s.pics_dirs)), p.stdin, protocol=4); p.stdin.close()
            while True:
                try: m = pickle.load(p.stdout)
                except EOFError: break
                if m[0] == 'prog': s._prog = '%s: %s, %.1f s' % (tag(m[1]), m[2], m[3]); print(s._prog, flush=True)
                elif m[0] == 'V':
                    it = items[m[1]]; info = m[2][1]; job['q'].append(('V', it, (m[2][0], VS.vscale(m[2][0]), it['cl'], info), m[3]))
                    print('%s: done: %d passes, converged %s, finite %.0f%%, solve8 %.2f s, %.1f s%s' % (tag(m[1]), info['passes'], info.get('converged'), 100 * info.get('finite', 0.), info['t'], m[3], s._qerr_text(info)), flush=True)
                elif m[0] == 'Tp': job['q'].append(('Tp', items[m[1]], m[2], m[3]))                  # пачка траекторий готова — окно дорисовывает сразу
                elif m[0] == 'T': job['q'].append(('T', items[m[1]], m[2], m[3])); print('%s: done: %s' % (tag(m[1]), s._traj_stat(m[2], m[3])), flush=True)
                elif m[0] == 'pics':
                    s._pics_note = '%s (%s)' % (s.pics_dirs[0] if s.pics_dirs else '?', time.strftime('%H:%M:%S')); print('[pics] %s -> %s  (%.1f s)' % (m[2], ', '.join(s.pics_dirs), m[3]), flush=True)
                elif m[0] == 'err': job['err'] = m[2]; print('[calc] stage failed: %s' % m[2], flush=True); break
            p.wait()
            if p.returncode and not job['err']: job['err'] = 'worker exit code %s' % p.returncode; print('[calc] stage failed: ' + job['err'], flush=True)
        except Exception as e:
            job['err'] = repr(e); print('[calc] stage failed: %s' % job['err'], flush=True)

    def _stage_adopt(s, ev):
        from . import vsurf as VS
        kind, it, payload, secs = ev; sig = it['sig']
        if kind == 'Tp':
            g_ = VS.traj_geom(payload, s.ax); acc = s._tpart.setdefault(sig, dict(parts=[], m=None)); acc['parts'].append(payload); acc['m'] = g_ if acc['m'] is None else {k: np.concatenate([acc['m'][k], g_[k]]) for k in g_}
            if sig == s._vsig: s._traj = (VS.merge_paths(acc['parts']), acc['m']); s._tprog = '%d/%d' % (sum(len(p['T']) for p in acc['parts']), VS.TRAJ_N); s._tfresh = True
            return
        if kind == 'V':
            vinfo = s._vstat(payload[3], sum(len(c) for c in it['cl'].values())); s._put_v(sig, (payload, vinfo))
            if s.runlog is not None: s.runlog.vcalc(it['calc'], payload[3], sum(len(c) for c in it['cl'].values()), secs)
        else:
            s._tpart.pop(sig, None); s._tprog = ''
            stat = s._traj_stat(payload, secs); s._put_t(sig, (payload, VS.traj_geom(payload, s.ax), stat))
            if s.runlog is not None: s.runlog.traj(it['calc'], payload, secs)
            if sig == s._vsig: s._traj = s._tcache[sig][:2]; s._tstat = stat; s._tfresh = True

    def stage_label(s):
        """строка статуса окна про этапы: идёт (прогресс) / что дальше"""
        j = s._stj
        if j is not None and j['th'].is_alive(): return s._prog
        return s._stinfo

    def _stage_text(s, snap):
        todo = s.stage_todo(snap)
        if not todo: return ''
        names = dict(s.VCALC); t = todo[0]
        return 'calc stages left %d: next %s %s - LMB one stage, RMB all' % (len(todo), names[t['calc']], 'V' if t['kind'] == 'V' else 'trajectories')

    def poll(s):
        """каждый кадр (main.update): готов ли фоновый расчёт (траектории по T или этапы) → в кэш и нарисовать"""
        sj = s._stj
        if sj is not None:
            ev = sj['q'][:]; del sj['q'][:len(ev)]
            for e in ev: s._stage_adopt(e)
            if not sj['th'].is_alive():
                s._stj = None; s._prog = ''
                if sj['err']: s._tfail = sj['items'][-1]['sig']
                ev2 = sj['q'][:]; del sj['q'][:]
                for e in ev2: s._stage_adopt(e)
                ev = ev + ev2
            if ev or s._stj is None: s._stinfo = s._stage_text(s.snap); s.draw(s.snap)
        j = s._tjob
        if j is None or j['th'].is_alive(): return
        s._tjob = None
        if j['err']: s._tstat = 'failed: ' + j['err']; print('[vsurf] trajectories ' + s._tstat, flush=True); s.draw(s.snap); return
        from . import vsurf as VS
        res = j['res']; stat = s._traj_stat(res, time.perf_counter() - j['t0'])
        s._put_t(j['sig'], (res, VS.traj_geom(res, s.ax), stat)); print('[vsurf] trajectories computed: ' + stat, flush=True)
        if s.runlog is not None: s.runlog.traj(j['sig'][0], res, time.perf_counter() - j['t0'])       # в кэш — даже если атлас/режим уже сменились (вернётся — не пересчитывать)
        if j['sig'] == s._vsig: s._traj = s._tcache[j['sig']][:2]; s._tstat = stat; s._tfresh = True
        s.draw(s.snap)

    def _box(s, extra):
        """граница поля ±(XLV + extra) по показанным осям — замкнутая ломаная (5, 3); для периодических осей (XLV = 0) — не рисуется стороной"""
        from ..algo import growN as G
        ex = np.broadcast_to(np.asarray(extra, float), (len(G.XLV),)); a, b = G.XLV[s.ax[0]] + ex[s.ax[0]], G.XLV[s.ax[1]] + ex[s.ax[1]]      # extra — число или вектор по осям (GM покомпонентно: рабочая зона ±(3, 2.5), буфер ±(6, 3.5))
        if G.XLV[s.ax[0]] == 0: a = SHOW_X if SHOW_X > 0 else np.pi                                # периодическая ось: рамка по периоду
        if G.XLV[s.ax[1]] == 0: b = np.pi
        C = np.array([(-a, -b), (a, -b), (a, b), (-a, b), (-a, -b)]); P = np.zeros((5, 3)); P[:, 0] = C[:, 0]; P[:, 1] = Y0; P[:, 2] = C[:, 1]; return P

    def _goal(s):
        """target outline as segments (K, 2, 3): circle/ellipse (GOALSHAPE='ball') or box (default), centre 0, half-widths RHOV along the shown axes."""
        if getattr(s, '_goal_segs', None) is None:
            from ..algo import growN as G
            n = len(G.RHOV); P = np.zeros((65, n)); ax = s.ax
            if getattr(G, 'GOALSHAPE', 'box') == 'ball':
                t = np.linspace(0, 2 * np.pi, 65); P[:, ax[0]] = G.RHOV[ax[0]] * np.cos(t); P[:, ax[1]] = G.RHOV[ax[1]] * np.sin(t)
            else:
                r0, r1 = G.RHOV[ax[0]], G.RHOV[ax[1]]; C = [(-r0, -r1), (r0, -r1), (r0, r1), (-r0, r1), (-r0, -r1)]; P = np.zeros((5, n))
                for i, (a_, b_) in enumerate(C): P[i, ax[0]] = a_; P[i, ax[1]] = b_
            Q = proj(P, ax); s._goal_segs = np.stack([Q[:-1], Q[1:]], 1)
        return s._goal_segs

    def retransform(s):
        """после зума/сдвига камеры ZoomManager сам вызывает apply_transform у зарегистрированных объектов — отдельно не нужно."""

    def pics_label(s):
        """строка статуса: куда пишутся картинки прогона (и когда обновлены)"""
        return ('pics: ' + (s._pics_note or ('%s (waiting for the first stage)' % s.pics_dirs[0]))) if s.pics_dirs else ''

    def field_text(s):
        """размеры поля в статусе: рабочая зона ±XLV и граница роста ±(XLV+GM) по осям x, v"""
        from ..algo import growN as G
        xl = np.asarray(G.XLV, float); gm = np.broadcast_to(np.asarray(getattr(G, 'GMV', getattr(G, 'GM', 0.)), float)[:len(xl)], xl.shape); return 'field: work zone x +-%g, v +-%g; growth limit x +-%g, v +-%g' % (xl[0], xl[1], xl[0] + gm[0], xl[1] + gm[1])

    def caption(s): return caption(s.snap)
