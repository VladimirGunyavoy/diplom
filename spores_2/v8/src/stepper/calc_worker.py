"""Счёт этапов V/траектории ВНЕ GIL окна (W1в, слово research-4 2026-10-12): отдельный процесс, запускается GrowView (growview._stage_run).
Вход (stdin, pickle): список (kind 'V'|'T', calc, cells_by_u). Выход (stdout, поток pickle): ('prog', i, текст, секунды) раз в ~1 с | ('V', i, (Vs, info без G), секунды) | ('T', i, res, секунды) | ('err', i, текст).
Окружение (M, KF, …) наследуется от окна. Печать библиотек уходит в stderr, stdout — только протокол."""
import os, sys, time, pickle
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = sys.stdout.buffer; sys.stdout = sys.stderr; sys.path.insert(0, V8)


def send(*m):
    pickle.dump(m, OUT, protocol=4); OUT.flush()


def draw(i, f, names):
    """картинки этапа (src/pics.py) в папки job['pics']; сбой картинки не валит расчёт"""
    t0 = time.perf_counter()
    try: f(); send('pics', i, names, time.perf_counter() - t0)
    except Exception as e:
        import traceback; traceback.print_exc(); send('prog', i, 'pics failed: %r' % (e,), 0.)


def main():
    from src.stepper import vsurf as VS
    from src import pics as PX
    job = pickle.load(sys.stdin.buffer); items = job['items']; dirs = job.get('pics') or []; Gby = {}; done_cells = [False]
    for i, (kind, calc, cl) in enumerate(items):
        t0 = time.perf_counter(); last = [t0]
        def say(msg, force=False):
            now = time.perf_counter()
            if force or now - last[0] >= 1.: last[0] = now; send('prog', i, msg, now - t0)
        try:
            say('start', True)
            if kind == 'V' or calc not in Gby:
                if kind == 'T': say('rebuilding V for trajectories', True)
                Vs, info = VS.solve_layers(cl, calc=calc, progress=lambda p, d, V: say('pass %d, residual %.1e, finite nodes %.0f%%' % (p, d, 100 * float((V < VS.BIG / 2).mean()))))
                Gby[calc] = info.pop('G')
                if kind == 'V': send('V', i, (Vs, info), time.perf_counter() - t0)
                if dirs: draw(i, lambda: ([] if done_cells[0] else PX.cells_png(Gby[calc], cl, dirs, title='(%s)' % os.environ.get('SYS', 'di'))) or (PX.error_map(Gby[calc], dirs, calc) if os.environ.get('SYS', 'di') == 'di' else None), 'cells.png, error_%s.png' % calc); done_cells[0] = True
            if kind == 'T':
                if getattr(Gby[calc], 'tab', 1) is None and hasattr(Gby[calc], 'build_table'): say('building the lookup table', True); Gby[calc].build_table(); say('lookup table %.1f s' % Gby[calc].t_table, True)     # таблица Q по слоям (b2): агенты берут lookup вместо поиска ячейки
                state = [0]
                def onb(part, done, total): state[0] = done; send('Tp', i, part, time.perf_counter() - t0); say('trajectories %d/%d' % (done, total), True)            # пачка готова — окно сразу дорисовывает
                res = VS.agent_paths(Gby[calc], cl=cl, batch=VS.TRAJ_BATCH, on_batch=onb, progress=lambda step, alive: say('trajectories %d/%d: agents in flight %d, step %d (t=%.2f)' % (state[0], VS.TRAJ_N, alive, step, step * .01)))
                send('T', i, res, time.perf_counter() - t0)
                if dirs: draw(i, lambda: (PX.traj_png(Gby[calc], res, dirs, calc), PX.hist_png(Gby[calc], res, dirs, calc)), 'traj_%s.png, hist_%s.png' % (calc, calc))
        except Exception as e:
            import traceback; traceback.print_exc(); send('err', i, repr(e), time.perf_counter() - t0); return


if __name__ == '__main__': main()
