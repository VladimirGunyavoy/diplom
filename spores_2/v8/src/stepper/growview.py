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
Y0 = .01


def proj(P, ax):
    """(..., n) -> (..., 3): (x, Y0, z) по осям ax."""
    P = np.asarray(P, float); out = np.empty(P.shape[:-1] + (3,)); out[..., 0] = P[..., ax[0]]; out[..., 1] = Y0; out[..., 2] = P[..., ax[1]]; return out


def cell_rows(G, ax):
    """отрезки строк клетки: (K, 2, 3) — между соседними узлами каждой строки."""
    P = proj(G, ax)                                   # (rows, M, 3)
    return np.stack([P[:, :-1], P[:, 1:]], 2).reshape(-1, 2, 3) if P.shape[1] > 1 else np.zeros((0, 2, 3))


def cell_outline(G, ax, halo=0.):
    """контур ядра (halo=0) или гало: ломаная по концам строк, концы строк сдвинуты наружу на halo вдоль строки; (K, 2, 3)."""
    P = proj(G, ax); a, b = P[:, 0].copy(), P[:, -1].copy()
    if halo:
        d = b - a; nrm = np.linalg.norm(d, axis=-1, keepdims=True); d = d / np.maximum(nrm, 1e-12); a, b = a - halo * d, b + halo * d
    ring = np.concatenate([a, b[::-1]], 0); ring = np.concatenate([ring, ring[:1]], 0)
    return np.stack([ring[:-1], ring[1:]], 1)


def build_geometry(snap, ax=(0, 1), halo=HALO):
    """dict имя слоя -> массив: 'points_*' (K, 3), 'segs_*' (K, 2, 3). Пустые слои — нулевой размер."""
    g = {}; cells = snap.get('cells') or []; cur = snap.get('cell')
    z3, z23 = np.zeros((0, 3)), np.zeros((0, 2, 3))
    g['points_done'] = np.concatenate([proj(c['G'], ax).reshape(-1, 3) for c in cells], 0) if cells else z3
    g['segs_rows_done'] = np.concatenate([cell_rows(c['G'], ax) for c in cells], 0) if cells else z23
    g['segs_core_done'] = np.concatenate([cell_outline(c['G'], ax) for c in cells], 0) if cells else z23
    g['segs_halo_done'] = np.concatenate([cell_outline(c['G'], ax, halo) for c in cells], 0) if cells else z23
    if cur is not None:
        g['points_cur'] = proj(cur['G'], ax).reshape(-1, 3); g['segs_rows_cur'] = cell_rows(cur['G'], ax)
        g['segs_core_cur'] = cell_outline(cur['G'], ax); g['segs_halo_cur'] = cell_outline(cur['G'], ax, halo)
    else: g['points_cur'], g['segs_rows_cur'], g['segs_core_cur'], g['segs_halo_cur'] = z3, z23, z23, z23
    sec = snap.get('section')                                          # пауза section: узлы базового сечения (K, n)
    if sec is not None: g['points_cur'] = np.concatenate([g['points_cur'], proj(sec, ax)], 0); sp = proj(sec, ax); g['segs_rows_cur'] = np.concatenate([g['segs_rows_cur'], np.stack([sp[:-1], sp[1:]], 1)], 0)
    seed = snap.get('seed'); g['points_seed'] = proj(np.asarray(seed, float)[None], ax).reshape(-1, 3) if seed is not None else z3
    q = snap.get('queue'); q = None if q is None else np.asarray(q, float); g['points_queue'] = proj(q.reshape(-1, q.shape[-1]), ax) if q is not None and q.size else z3
    return g


def caption(snap):
    """одна строка подписи: фаза, слой, причина стопа."""
    if snap is None: return 'нет снимка (N — шаг, G — затравка в точке взгляда)'
    r = snap.get('reason'); cells = snap.get('cells') or []
    return 'фаза: %s  (ур. %s)   слой u=%s   клеток: %d   очередь: %d%s' % (snap.get('phase'), snap.get('lvl'), snap.get('u'), len(cells), len(snap.get('queue') if snap.get('queue') is not None else []), ('   СТОП: %s' % r) if r else '')


class TimeTable:
    """время по именам пауз: вычисления (из stepper.times) и отрисовка (здесь)."""
    def __init__(s): s.draw = {}; s.n = {}
    def add_draw(s, phase, dt): s.draw[phase] = s.draw.get(phase, 0.) + dt; s.n[phase] = s.n.get(phase, 0) + 1
    def text(s, compute=None, top=8):
        compute = compute or {}; names = sorted(set(compute) | set(s.draw), key=lambda k: -(compute.get(k, 0.) + s.draw.get(k, 0.)))[:top]
        rows = ['%-18s %9s %9s %4s' % ('пауза', 'вычисл,мс', 'рисов,мс', 'раз')] + ['%-18s %9.1f %9.1f %4d' % (k, 1e3 * compute.get(k, 0.), 1e3 * s.draw.get(k, 0.), s.n.get(k, 0)) for k in names]
        return '\n'.join(rows)


class GrowView:
    """Ursina-часть: создаёт по Scalable на слой геометрии и перерисовывает при смене снимка. Импорт Ursina — лениво."""
    COL = {'points_done': (0.55, 0.75, 1.0, .8), 'segs_rows_done': (0.4, 0.55, 0.9, .5), 'segs_core_done': (1.0, 1.0, 1.0, .9), 'segs_halo_done': (0.6, 0.6, 0.6, .4),
           'points_cur': (1.0, .8, .2, 1), 'segs_rows_cur': (1.0, .6, .1, .9), 'segs_core_cur': (1.0, .3, .1, 1), 'segs_halo_cur': (1.0, .5, .3, .5),
           'points_seed': (.2, 1.0, .3, 1), 'points_queue': (.9, .4, 1.0, .8)}

    def __init__(s, zoom_manager, ax=(0, 1), halo=HALO):
        s.zm = zoom_manager; s.ax = ax; s.halo = halo; s.layers = {}; s.last = None; s.times = TimeTable(); s.snap = None

    def _layer(s, name):
        if name not in s.layers:
            from ursina import Mesh, Vec3
            from ..core.scalable import Scalable
            pts = name.startswith('points'); mode = 'point' if pts else 'line'
            col = s.COL[name]

            class _Multi(Scalable):
                def __init__(self):
                    self.real_v = np.zeros((0, 3)); super().__init__(model=Mesh(vertices=[], mode=mode, thickness=(.1 if name in ('points_seed', 'points_cur') else .05) if pts else 2))
                def apply_transform(self, a, b, **kw):
                    v = self.real_v * a + b; self.model.vertices = [Vec3(*p) for p in v]; self.model.generate()
            e = _Multi(); from ursina import color; e.color = color.rgba(*col); e.alpha = col[3]
            s.zm.register_object(e, name='growview_' + name); s.layers[name] = e
        return s.layers[name]

    def draw(s, snap):
        """нарисовать снимок (None — очистить). Время отрисовки уходит в TimeTable по имени паузы."""
        t0 = time.perf_counter(); s.snap = snap
        g = build_geometry(snap, s.ax, s.halo) if snap is not None else {k: np.zeros((0, 3)) for k in s.COL}
        for name in s.COL:
            arr = g.get(name, np.zeros((0, 3))); e = s._layer(name); e.real_v = arr.reshape(-1, 3); e.apply_transform(s.zm.a_transformation, s.zm.b_translation)
        if snap is not None: s.times.add_draw(snap.get('phase', '?'), time.perf_counter() - t0)

    def retransform(s):
        """после зума/сдвига камеры ZoomManager сам вызывает apply_transform у зарегистрированных объектов — отдельно не нужно."""

    def caption(s): return caption(s.snap)
