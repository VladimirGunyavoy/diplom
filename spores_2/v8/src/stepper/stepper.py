"""Пошаговый просмотр алгоритма (PLAN п.49, вариант А). Алгоритм идёт в фоновом потоке и зовёт pause(name, lvl, **state).
Без установленного Stepper pause — пустая (один `is None`). Уровни: 1 — крупные, 3 — мелкие (чем меньше, тем крупнее).
Клавиши (главный поток зовёт cmd): N — до следующей включённой паузы; M — до паузы того же или крупнее уровня (lvl <= текущего);
C — до конца или до обязательной (must=True). Значения state могут быть callable без аргументов — их зовут только если пауза реально стоит.
Снимок — копия: массивы numpy копируются, кроме read-only (их считаем неизменными и не копируем — готовые клетки)."""
import threading, time, copy
import numpy as np

ACTIVE = None                                    # установленный Stepper или None


class Aborted(BaseException):
    """поднимается в потоке алгоритма при Stepper.abort()"""


def pause(name, lvl=2, must=False, **state):
    s = ACTIVE
    if s is not None: s._pause(name, lvl, must, state)


def _copy(x):
    if callable(x): x = x()
    if isinstance(x, np.ndarray): return x if not x.flags.writeable else x.copy()
    if isinstance(x, dict): return {k: _copy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_copy(v) for v in x)
    return copy.copy(x)


class Stepper:
    """cfg: max_lvl — паузы глубже (lvl > max_lvl) пропускаются; off — имена выключенных пауз; on — если задано, только эти имена."""
    def __init__(s, max_lvl=3, off=(), on=None, start_paused=True, log=None):
        s.max_lvl, s.off, s.on = max_lvl, set(off), (set(on) if on else None); s.log = log
        s.mode = 'n' if start_paused else 'c'; s.ref_lvl = 3
        s._go = threading.Event(); s._stopped = threading.Event(); s._lock = threading.Lock()
        s.snap = None; s.version = 0; s.finished = False; s.result = None; s.error = None
        s.times = {}; s.counts = {}; s.render_time = 0.; s.render_n = 0; s.history = []     # times[name] — вычисления перед паузой name; history — (№, name, lvl, dt)
        s._mark = None; s._abort = False; s._th = None; s._k = 0

    def enabled(s, name, lvl): return lvl <= s.max_lvl and name not in s.off and (s.on is None or name in s.on)

    # --- поток алгоритма ---
    def start(s, fn, *a, **kw):
        global ACTIVE
        ACTIVE = s; s._th = threading.Thread(target=s._run, args=(fn, a, kw), daemon=True); s._mark = time.perf_counter(); s._th.start(); return s

    def _run(s, fn, a, kw):
        global ACTIVE
        try: s.result = fn(*a, **kw)
        except Aborted: pass
        except BaseException as e: s.error = e
        finally:
            s.finished = True; s._stopped.set()
            if ACTIVE is s: ACTIVE = None

    def _pause(s, name, lvl, must, state):
        if threading.current_thread() is not s._th: return          # зовут не из потока алгоритма — не трогаем
        if not (must or s.enabled(name, lvl)): return
        now = time.perf_counter(); dt = now - s._mark; s.times[name] = s.times.get(name, 0.) + dt; s.counts[name] = s.counts.get(name, 0) + 1; s._k += 1
        if s.log: s.log('%d %s lvl%d %.4f с' % (s._k, name, lvl, dt))
        s.history.append((s._k, name, lvl, dt))
        stop = must or s.mode == 'n' or (s.mode == 'm' and lvl <= s.ref_lvl)
        if stop:
            snap = dict(phase=name, lvl=lvl, must=must, step=s._k, dt=dt, **{k: _copy(v) for k, v in state.items()})
            with s._lock: s.snap = snap; s.version += 1
            s._go.clear(); s._stopped.set(); s._go.wait()
            if s._abort: raise Aborted()
        s._mark = time.perf_counter()

    # --- главный поток ---
    def cmd(s, key):
        """'n' | 'm' | 'c'. Возвращает False, если алгоритм не стоит на паузе (команда проигнорирована)."""
        if s.finished or not s._stopped.is_set(): return False
        s.mode = key
        if key == 'm' and s.snap: s.ref_lvl = s.snap['lvl']
        s._stopped.clear(); s._go.set(); return True

    def poll(s, since=-1):
        """новый снимок (если version > since) → (version, snap) или None"""
        with s._lock:
            return (s.version, s.snap) if s.version > since and s.snap is not None else None

    def wait_paused(s, timeout=10.):
        """до остановки на паузе или конца алгоритма; True — стоит на паузе"""
        s._stopped.wait(timeout); return not s.finished and s._stopped.is_set()

    def note_render(s, dt): s.render_time += dt; s.render_n += 1

    def table(s):
        """строки (имя, вызовов, всего с, среднее мс) по убыванию времени + строка отрисовки"""
        rows = sorted(((n, s.counts[n], t, 1000 * t / s.counts[n]) for n, t in s.times.items()), key=lambda r: -r[2])
        if s.render_n: rows.append(('render (гл. поток)', s.render_n, s.render_time, 1000 * s.render_time / s.render_n))
        return rows

    def abort(s):
        s._abort = True; s._go.set()
        if s._th: s._th.join(2.)
