"""DemoStepper — синтетические снимки роста клеток ДИ 2D (x, v) по контракту п.49, пока нет настоящего stepper (b6).
Клетка слоя u: строки — точки вдоль потока (x = x0 + v0 t + u t²/2, v = v0 + u t), M узлов поперёк (сдвиг по x). Интерфейс (как ждёт main.py):
  snapshot -> dict | None; times -> {имя паузы: секунды вычислений}; key('N'|'M'|'C'); seed_at(xv) — новая затравка."""
import time
import numpy as np


def _cell(p, u, T=.6, rows=7, M=5, r=.12, tr=1.0):
    t = np.linspace(0, T, rows)[:, None]; s = np.linspace(-r, r, M)[None, :]
    x = p[0] + p[1] * t + u * t ** 2 / 2 + s; v = p[1] + u * t + 0 * s
    return dict(G=np.stack([x, v], -1), c=np.array(p, float), r=r, e=np.array([[1., 0.]]), nb=0, nf=rows)


def gen(seed, u=1.0, ncells=6):
    """генератор (имя паузы, lvl, состояние); каждая пауза = точка `pause(...)` алгоритма."""
    cells = []; queue = [np.array(seed, float)]
    yield 'seed', 1, dict(u=u, seed=queue[0].copy(), cell=None, cells=[], reason=None, queue=np.array(queue))
    for k in range(ncells):
        p = queue.pop(0); c = _cell(p, u); time.sleep(.02)
        yield 'section', 2, dict(u=u, seed=p.copy(), cell=dict(G=c['G'][:1], r=c['r']), cells=list(cells), reason=None, queue=np.array(queue) if queue else np.zeros((0, 2)))
        for i in range(2, c['G'].shape[0] + 1):
            time.sleep(.01); yield 'row_fwd', 3, dict(u=u, seed=p.copy(), cell=dict(G=c['G'][:i], r=c['r']), cells=list(cells), reason=None, queue=np.array(queue) if queue else np.zeros((0, 2)))
        yield 'side', 3, dict(u=u, seed=p.copy(), cell=c, cells=list(cells), reason=None, queue=np.array(queue) if queue else np.zeros((0, 2)))
        cells.append(c); new = c['G'][-1, -1] + np.array([.02, 0.]); queue.append(new); queue.append(c['G'][len(c['G']) // 2, 0] - np.array([.3, 0.]))
        yield 'stop', 2, dict(u=u, seed=p.copy(), cell=c, cells=list(cells), reason=['изгиб', 'сосед', 'TMAX', 'поле', 'цель', 'RMAX'][k % 6], queue=np.array(queue))
        yield 'cell_done', 1, dict(u=u, seed=p.copy(), cell=None, cells=list(cells), reason=None, queue=np.array(queue))


class DemoStepper:
    def __init__(s, seed=(0., 0.), u=1.0):
        s.u = u; s.reset(seed); s.times = {}

    def reset(s, seed):
        s.g = gen(seed, s.u); s.snapshot = None; s.lvl = 9; s.done = False; s.times = {}

    def seed_at(s, xv): s.reset(tuple(map(float, xv)))

    def _step(s):
        t0 = time.perf_counter()
        try: name, lvl, st = next(s.g)
        except StopIteration: s.done = True; return False
        s.times[name] = s.times.get(name, 0.) + time.perf_counter() - t0; s.lvl = lvl; s.snapshot = dict(phase=name, lvl=lvl, **st); return True

    def key(s, k):
        """N — следующая пауза; M — до паузы того же или крупнее (lvl <= текущий); C — до конца."""
        k = k.upper()
        if k == 'N': s._step()
        elif k == 'M':
            cur = s.snapshot['lvl'] if s.snapshot else 9
            while s._step():
                if s.snapshot['lvl'] <= cur: break
        elif k == 'C':
            while s._step(): pass
