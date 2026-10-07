"""LiveStepper — настоящий рост ДИ 2D (v8/src/algo/growN.py, SYS=di) под Stepper (b6) для main.py; тот же интерфейс, что у DemoStepper:
snapshot (последний, с переносом cells/queue/seed между снимками), times, key('N'|'M'|'C'), seed_at(xv)."""
import os, threading
import sys
for _s in (sys.stdout, sys.stderr):                           # growN печатает по-русски; на cp1252-консоли print в потоке алгоритма падал и слой «кончался» с error
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
import numpy as np
for k, v in dict(SYS='di', M='3', KF='21', MAXC='60', NFAIL='400', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000', GS='0', GLIM='0', GSEED='1', GOALB='0', GOALSHAPE='ball', RHO='.2').items(): os.environ.setdefault(k, v)   # настройки показа; свои — переменными окружения
from ..algo import growN as g
from . import stepper as S

LAYERS = tuple(getattr(g, 'ULAYERS', g.US))             # порядок слоёв: для di (1, -1, 0) — берётся из growN


class LiveStepper:
    def __init__(s, seed=None, u=None, max_lvl=3, layers=None):
        s.layers = tuple(layers) if layers is not None else LAYERS if u is None else (u,); s.u = s.layers[0]; s.max_lvl = max_lvl; s.cells_by_layer = {}; s._st = None; s.snapshot = None; s._ver = -1; s._carry = {}; s.history = []; s._tm = []; s.pos = -1; s.depth = 3; s.seed_at(seed)

    @property
    def times(s):
        if s.viewing_history: return dict(s._tm[s.pos])                    # table as it was at that snapshot
        return dict(s._st.times) if s._st else {}

    @property
    def viewing_history(s): return 0 <= s.pos < len(s.history) - 1

    @property
    def hist_info(s):
        """(k, K) when a past snapshot is shown, else None"""
        return (s.pos + 1, len(s.history)) if s.viewing_history else None

    def _push(s, sn):
        s.history.append(sn); s._tm.append(dict(s._st.times)); s.pos = len(s.history) - 1; s.snapshot = sn

    def back(s):
        """one snapshot back (Z); False if already at the first one"""
        if s.pos <= 0: return False
        s.pos -= 1; s.snapshot = s.history[s.pos]; return True

    def _live(s): s.pos = len(s.history) - 1; s.snapshot = s.history[s.pos] if s.history else None

    def _layer(s, seed):
        """фоновая функция: слои по порядку; у каждого свой idx (build_layer) и свой rng; ручная затравка — только у первого слоя, дальше затравки выбирает алгоритм;
        между слоями пауза 'layer' (lvl 0): все клетки слоя; результат — dict u -> [cell_dict]"""
        out = {}
        for k, u in enumerate(s.layers):
            sd = seed if k == 0 else None
            res = g.build_layer(u, np.random.default_rng(k), seeds=None if sd is None else [np.asarray(sd, float)], only_queue=sd is not None)[0]
            cells = [g.cell_dict(c) for c in res]; out[u] = cells
            S.pause('layer', 0, u=u, layer_k=k, cells=cells, queue=[], reason='layer u=%s finished: %d cells (%d/%d)' % (u, len(cells), k + 1, len(s.layers)))
        return out    # seed=None — затравки выбирает сам алгоритм

    def seed_at(s, xv):
        if s._st is not None: s._st.abort()
        s.snapshot = None; s._ver = -1; s._carry = {}; s.cells_by_layer = {}; s.history = []; s._tm = []; s.pos = -1; seed = None if xv is None else tuple(float(x) for x in xv)
        s._st = S.Stepper(max_lvl=s.max_lvl).start(lambda: s._layer(seed)); s._st.wait_paused(10.); s._pull()    # первая пауза (посев) сразу

    def _pull(s):
        r = s._st.poll(s._ver)
        if r is None: return
        s._ver, sn = r; sn = dict(sn)
        for k in ('cells', 'queue'):                                  # приходят только в паузах seed/cell — переносим дальше
            if k in sn: s._carry[k] = sn[k]
            else: sn[k] = s._carry.get(k)
        u = sn.get('u'); kk = s.layers.index(u) if u in s.layers else 0
        if sn.get('phase') == 'layer': s.cells_by_layer[u] = sn['cells']
        sn['layers'] = s.layers; sn['layer_k'] = kk; sn['layers_n'] = len(s.layers); sn['cells_by_layer'] = {uu: c for uu, c in s.cells_by_layer.items() if s.layers.index(uu) < kk}      # прошлые слои (текущий — в 'cells')
        s._push(sn)

    def key(s, k):
        k = k.lower()
        if s.viewing_history:                                          # after Z: N goes forward through the history, M/C first catch up to the live snapshot
            if k == 'n': s.pos += 1; s.snapshot = s.history[s.pos]; return
            s._live()
        ref = None
        if k == 'n':                                                   # N / LMB: next pause with lvl <= depth (3 rows, 2 stages, 1 spores, 0 to the end of the layer)
            if s.depth <= 0: k, ref = 'm', 0
            elif s.depth < 3: k, ref = 'm', s.depth
        if s._st.cmd(k, ref): s._st.wait_paused(30.); s._pull(); s._final()

    def _final(s):
        """алгоритм закончил все слои (C или конец последнего) — финальный снимок: клетки последнего слоя в 'cells', остальные в cells_by_layer."""
        if s._st.finished and isinstance(s._st.result, dict) and (s.snapshot is None or s.snapshot.get('phase') != 'done'):
            res = s._st.result; s.cells_by_layer = dict(res); kk = len(s.layers) - 1; ul = s.layers[-1]; sn = dict(s.snapshot or {})
            sn.update(phase='done', lvl=0, cell=None, cells=res.get(ul, []), queue=[], u=ul, layers=s.layers, layer_k=kk, layers_n=len(s.layers), cells_by_layer={uu: c for uu, c in res.items() if uu != ul},
                      reason='all layers finished: ' + ', '.join('u=%s: %d' % (uu, len(c)) for uu, c in res.items())); s._push(sn)

    @property
    def done(s): return s._st.finished

    DEPTH_NAMES = {3: 'rows', 2: 'stages', 1: 'spores', 0: 'layer'}

    @property
    def depth_name(s): return s.DEPTH_NAMES.get(s.depth, str(s.depth))

    def up(s):
        """RMB: one level up. Inside a spore (lvl >= 2) -> to the nearest lvl-1 pause (cell/seed), depth = 1; on seed/cell -> to the end of the layer, depth = 0."""
        if s.viewing_history: s._live()
        lvl = (s.snapshot or {}).get('lvl')
        if lvl is not None and lvl >= 2: s.depth = 1; s._go('m', 1)
        else: s.depth = 0; s._go('m', 0)

    def finer(s):
        """Shift+LMB: depth + 1 and a step"""
        s.depth = min(3, s.depth + 1); s.key('n')

    def _go(s, k, ref=None):
        if s._st.cmd(k, ref): s._st.wait_paused(30.); s._pull(); s._final()
