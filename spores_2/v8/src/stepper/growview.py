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
import time
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


def build_strips(snap, ax=(0, 1)):
    """линии слоёв как ломаные (по одной на клетку): строки — зигзаг по узлам подряд (L₋₁→C₋₁→R₋₁→L₀→…), контур ядра и гало — замкнутая ломаная; dict имя слоя -> список (n, 3)"""
    out = {k: [] for k in ('segs_rows_done', 'segs_center_done', 'segs_core_done', 'segs_halo_done', 'segs_rows_cur', 'segs_center_cur', 'segs_core_cur', 'segs_halo_cur')}
    def ring(o): return np.concatenate([o[:, 0], o[-1:, 1]], 0)
    def cell(G, suf):
        P = proj(G, ax)
        if P.shape[1] > 1: out['segs_rows_' + suf].append(P.reshape(-1, 3))
        if P.shape[0] > 1: out['segs_center_' + suf].append(P[:, P.shape[1] // 2])        # траектория зерна: центральные узлы строк
        out['segs_core_' + suf].append(ring(cell_outline(G, ax))); out['segs_halo_' + suf].append(ring(cell_outline(G, ax, True)))
    for c in snap.get('cells') or []: cell(c['G'], 'done')
    cur = snap.get('cell')
    if cur is not None: cell(cur['G'], 'cur')
    sec = snap.get('section')
    if sec is not None: out['segs_rows_cur'].append(proj(sec, ax))
    return out


def seg_index(K):
    """индексы отрезков для Mesh(mode='line'): без triangles ursina рисует ОДНУ ломаную через все вершины, а у нас вершины — пары [a0, b0, a1, b1, ...]."""
    return [(2 * i, 2 * i + 1) for i in range(K)]


def build_geometry(snap, ax=(0, 1), halo=HALO):
    """dict имя слоя -> массив: 'points_*' (K, 3), 'segs_*' (K, 2, 3). Пустые слои — нулевой размер."""
    g = {}; cells = snap.get('cells') or []; cur = snap.get('cell')
    z3, z23 = np.zeros((0, 3)), np.zeros((0, 2, 3))
    g['points_done'] = np.concatenate([proj(c['G'], ax).reshape(-1, 3) for c in cells], 0) if cells else z3
    g['segs_rows_done'] = np.concatenate([cell_rows(c['G'], ax) for c in cells], 0) if cells else z23
    g['segs_core_done'] = np.concatenate([cell_outline(c['G'], ax) for c in cells], 0) if cells else z23
    g['segs_halo_done'] = np.concatenate([cell_outline(c['G'], ax, True) for c in cells], 0) if cells else z23
    if cur is not None:
        g['points_cur'] = proj(cur['G'], ax).reshape(-1, 3); g['segs_rows_cur'] = cell_rows(cur['G'], ax)
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


def _discs(P, R, res=16):
    """K центров (K, 3) → вершины (K·(res+1), 3) и индексы треугольников: плоские диски радиуса R в плоскости XZ (как Spore: Circle, rotation 90°)"""
    K = len(P)
    if K == 0: return np.zeros((0, 3), np.float32), np.zeros((0,), np.uint32)
    t = np.linspace(0, 2 * np.pi, res, endpoint=False); rim = np.stack([R * np.cos(t), np.zeros(res), R * np.sin(t)], 1)
    V = np.concatenate([P[:, None, :], P[:, None, :] + rim[None]], 1).reshape(-1, 3).astype(np.float32)
    tri = np.stack([np.zeros(res, np.int64), 1 + np.arange(res), 1 + (np.arange(res) + 1) % res], 1)
    idx = (np.arange(K)[:, None, None] * (res + 1) + tri[None]).reshape(-1).astype(np.uint32)
    return V, idx


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
            super().__init__(); self._gn = None; self.col = (1., 1., 1., 1.); self.real_position = np.zeros(3); self.real_scale = np.ones(3)
        def set_geometry(self, V, idx=None, kind='lines', thick=None):
            if self._gn is not None: self._gn.remove_node(); self._gn = None
            if len(V) == 0: return
            self._gn = self.attach_new_node(_geom_node(V, kind, idx, self.col)); self._gn.set_transparency(TransparencyAttrib.M_alpha)
            if kind != 'tris': self._gn.set_render_mode_thickness(thick or 2)
            else: self._gn.set_two_sided(True)
            self._gn.set_depth_offset(1)                                   # поверх пола (y = 0), без z-fighting
        def apply_transform(self, a, b, **kw): self.position = b; self.scale = a
    return _Batch


def _Batch():
    global _Batch
    _Batch = _make_batch(); return _Batch()


class GrowView:
    """Ursina-часть: создаёт по Scalable на слой геометрии и перерисовывает при смене снимка. Импорт Ursina — лениво."""
    # цвета: центральная линия + зерно — зелёные; боковые линии — оранжевые (гало бледнее ядра); зигзаг строк и их узлы — голубые; текущая клетка ярче, готовые приглушены
    COL = {'points_done': (.50, .70, .90, .9), 'segs_rows_done': (.40, .62, .82, .6), 'segs_center_done': (.35, .78, .45, .9), 'segs_core_done': (.88, .60, .30, .9), 'segs_halo_done': (.72, .55, .38, .45),
           'points_cur': (.30, .85, 1.0, 1), 'segs_rows_cur': (.30, .80, 1.0, .9), 'segs_center_cur': (.20, 1.0, .30, 1), 'segs_core_cur': (1.0, .55, .10, 1), 'segs_halo_cur': (1.0, .70, .40, .6),
           'points_seed': (.2, 1.0, .3, 1), 'points_queue': (.9, .4, 1.0, .8), 'segs_goal': (.55, 1.0, .65, .9)}

    def __init__(s, zoom_manager, ax=(0, 1), halo=HALO, spore_manager=None):
        s.sm = spore_manager; s._pts = {}; s._built_size = None; s.zm = zoom_manager; s.ax = ax; s.halo = halo; s.layers = {}; s.last = None; s.times = TimeTable(); s.snap = None

    def _layer(s, name):
        """один объект Ursina на слой (один GeomNode, один примитив); зум/сдвиг — трансформ узла, вершины в реальных координатах"""
        if name not in s.layers:
            e = _Batch(); e.col = tuple(float(x) for x in s.COL[name])
            s.zm.register_object(e, name='growview_' + name); s.layers[name] = e
        return s.layers[name]

    def _set_layer(s, name, arr):
        e = s._layer(name); arr = np.asarray(arr, float).reshape(-1, 3)
        if name.startswith('points'):
            s._pts[name] = arr; R = .5 * SPORE_SCALE * (s.sm.size if s.sm is not None else 1.); s._built_size = None if s.sm is None else s.sm.size    # диски как у Spore (Circle r = .5·size), в мире
            e.set_geometry(*_discs(arr, R), kind='tris')
        else: e.set_geometry(arr, kind='lines', thick=2)

    def _set_strips(s, name, strips):
        e = s._layer(name); strips = [np.asarray(x, float).reshape(-1, 3) for x in strips if len(x) > 1]
        if not strips: e.set_geometry(np.zeros((0, 3))); return
        starts = np.cumsum([0] + [len(x) for x in strips[:-1]]); e.set_geometry(np.concatenate(strips, 0), [(int(a), len(x)) for a, x in zip(starts, strips)], kind='strips', thick=2)

    def refresh(s):
        """размер спор ('1') поменялся — пересобрать диски (зум этого не требует)"""
        if s.sm is not None and getattr(s, '_built_size', None) not in (None, s.sm.size):
            for name, arr in list(s._pts.items()): s._set_layer(name, arr)

    def draw(s, snap):
        """нарисовать снимок (None — очистить). Время отрисовки уходит в TimeTable по имени паузы."""
        t0 = time.perf_counter(); s.snap = snap
        if snap is None: s.draw_hist = {}
        elif id(snap) in getattr(s, 'draw_hist', {}):                  # a past snapshot (Z): draw it again, restore the draw-time table as it was
            s.times.draw, s.times.n = (dict(x) for x in s.draw_hist[id(snap)]); t0 = None
        g = build_geometry(snap, s.ax, s.halo) if snap is not None else {k: np.zeros((0, 3)) for k in s.COL}
        g['segs_goal'] = s._goal()                                      # target set: static, stays when the scene is cleared
        st = build_strips(snap, s.ax) if snap is not None else {}
        gs = s._goal(); st['segs_goal'] = [np.concatenate([gs[:, 0], gs[-1:, 1]], 0)]                      # цель — замкнутая ломаная
        for name in s.COL:
            if name.startswith('points'): s._set_layer(name, g.get(name, np.zeros((0, 3))))
            else: s._set_strips(name, st.get(name, []))
        if snap is not None and t0 is not None:
            s.times.add_draw(snap.get('phase', '?'), time.perf_counter() - t0); s.draw_hist = getattr(s, 'draw_hist', {}); s.draw_hist[id(snap)] = (dict(s.times.draw), dict(s.times.n))

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

    def caption(s): return caption(s.snap)
