"""Счёт этапов V/траектории ВНЕ GIL окна (W1в, слово research-4 2026-10-12): отдельный процесс, запускается GrowView (growview._stage_run).
Вход (stdin, pickle): список (kind 'V'|'T', calc, cells_by_u). Выход (stdout, поток pickle): ('prog', i, текст, секунды) раз в ~1 с | ('V', i, (Vs, info без G), секунды) | ('T', i, res, секунды) | ('err', i, текст).
Окружение (M, KF, …) наследуется от окна. Печать библиотек уходит в stderr, stdout — только протокол."""
import os, sys, time, pickle
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = sys.stdout.buffer; sys.stdout = sys.stderr; sys.path.insert(0, V8)


def send(*m):
    pickle.dump(m, OUT, protocol=4); OUT.flush()


def main():
    from src.stepper import vsurf as VS
    items = pickle.load(sys.stdin.buffer); Gby = {}
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
            if kind == 'T':
                res = VS.agent_paths(Gby[calc], progress=lambda step, alive: say('agents in flight %d of %d, step %d (t=%.2f)' % (alive, VS.TRAJ_N, step, step * .01)))
                send('T', i, res, time.perf_counter() - t0)
        except Exception as e:
            import traceback; traceback.print_exc(); send('err', i, repr(e), time.perf_counter() - t0); return


if __name__ == '__main__': main()
