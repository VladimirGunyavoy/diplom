"""LiveStepper — настоящий рост ДИ 2D (v8/src/algo/growN.py, SYS=di) под Stepper (b6) для main.py; тот же интерфейс, что у DemoStepper:
snapshot (последний, с переносом cells/queue/seed между снимками), times, key('N'|'M'|'C'), seed_at(xv)."""
import os, threading
import numpy as np
for k, v in dict(SYS='di', M='3', KF='21', MAXC='60', NFAIL='40', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000').items(): os.environ.setdefault(k, v)   # настройки показа; свои — переменными окружения
from ..algo import growN as g
from . import stepper as S


class LiveStepper:
    def __init__(s, seed=(0., 0.), u=None, max_lvl=3):
        s.u = g.US[1] if u is None else u; s.max_lvl = max_lvl; s._st = None; s.snapshot = None; s._ver = -1; s._carry = {}; s.seed_at(seed)

    @property
    def times(s): return dict(s._st.times) if s._st else {}

    def _layer(s, seed):
        return g.build_layer(s.u, np.random.default_rng(0), seeds=[np.asarray(seed, float)], only_queue=True)[0]

    def seed_at(s, xv):
        if s._st is not None: s._st.abort()
        s.snapshot = None; s._ver = -1; s._carry = {}; seed = tuple(float(x) for x in xv)
        s._st = S.Stepper(max_lvl=s.max_lvl).start(lambda: s._layer(seed)); s._st.wait_paused(10.); s._pull()    # первая пауза (посев) сразу

    def _pull(s):
        r = s._st.poll(s._ver)
        if r is None: return
        s._ver, sn = r; sn = dict(sn)
        for k in ('cells', 'queue'):                                  # приходят только в паузах seed/cell — переносим дальше
            if k in sn: s._carry[k] = sn[k]
            else: sn[k] = s._carry.get(k)
        s.snapshot = sn

    def key(s, k):
        k = k.lower()
        if s._st.cmd(k): s._st.wait_paused(30.); s._pull(); s._final()

    def _final(s):
        """алгоритм закончил (C или конец слоя) — финальный снимок из результата build_layer: все клетки слоя, причина — конец."""
        if s._st.finished and s._st.result is not None and (s.snapshot is None or s.snapshot.get('phase') != 'done'):
            cells = [g.cell_dict(c) for c in s._st.result]; sn = dict(s.snapshot or {}); sn.update(phase='done', lvl=0, cell=None, cells=cells, queue=sn.get('queue'), reason='layer finished: %d cells' % len(cells)); s.snapshot = sn

    @property
    def done(s): return s._st.finished
