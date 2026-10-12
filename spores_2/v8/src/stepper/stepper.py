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
    def __init__(s, max_lvl=3, off=(), on=None, start_paused=True, log=None, log_stops=True, log_actions=True):
        s.max_lvl, s.off, s.on = max_lvl, set(off), (set(on) if on else None); s.log = log
        s.log_stops, s.log_actions, s.echo = log_stops, log_actions, lambda m: print(m, flush=True); s._spore = 0; s._in_spore = False; s._was_in = False; s._stops = {}       # консольный лог: echo(строка); _spore — счётчик пауз seed
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
        try:
            s.result = fn(*a, **kw)
            if s.log_actions: r = s.result; s.echo('layer finished: %d cells' % len(r[0] if isinstance(r, tuple) else r))
        except Aborted: pass
        except BaseException as e: s.error = e
        finally:
            s.finished = True; s._stopped.set()
            if ACTIVE is s: ACTIVE = None

    def _pause(s, name, lvl, must, state):
        if threading.current_thread() is not s._th: return          # зовут не из потока алгоритма — не трогаем
        if s.log_stops or s.log_actions: s._note(name, state)               # лог причин — даже если пауза пропущена (без вызова callable-полей, кроме cell на итоге)
        if not (must or s.enabled(name, lvl)): return
        now = time.perf_counter(); dt = now - s._mark; s.times[name] = s.times.get(name, 0.) + dt; s.counts[name] = s.counts.get(name, 0) + 1; s._k += 1
        if s.log: s.log('%d %s lvl%d %.4f s' % (s._k, name, lvl, dt))
        s.history.append((s._k, name, lvl, dt))
        stop = must or s.mode == 'n' or (s.mode == 'm' and lvl <= s.ref_lvl)
        if stop:
            snap = dict(phase=name, lvl=lvl, must=must, step=s._k, dt=dt, **{k: _copy(v) for k, v in state.items()})
            with s._lock: s.snap = snap; s.version += 1
            if s.log_actions: s._say(name, snap)
            s._go.clear(); s._stopped.set(); s._go.wait()
            if s._abort: raise Aborted()
        s._mark = time.perf_counter()

    # --- консольный лог ---
    @staticmethod
    def _dir(d):
        return 'forward' if d[0] == 'F' else 'back' if d[0] == 'B' else 'side%s%d' % ('+' if d[2] > 0 else '-', d[1])

    def _note(s, name, state):
        try:
            if name == 'seed': s._spore += 1; s._in_spore = True; s._stops = {}
            elif name == 'stop':
                d = s._dir(state['dir']); s._stops[d] = state['reason']
                if s.log_stops: s.echo('[stop] spore #%d %s reason=%s' % (s._spore, d, state['reason']))
            elif name == 'reject': s._was_in = s._in_spore; s._in_spore = False
            elif name == 'cell':
                s._in_spore = False
                if s.log_stops:
                    c = state['cell']; c = c() if callable(c) else c
                    s.echo('[cell] spore #%d rows %d+%d (+%d/%d halo) r=%s stops: %s' % (s._spore, c['nb'], c['nf'], c['hb'], c['hf'], np.round(c['r'], 3).tolist(), s._stops))
        except Exception as e: s.echo('[stepper] log error: %r' % (e,))

    def _say(s, name, snap):
        """строка человеческим языком про реально стоящую паузу (snap — уже копия)"""
        k = s._spore; c = snap.get('cell')
        try:
            if name == 'seed': m = 'spore #%d: seed at (%s)' % (k, ', '.join('%.3f' % x for x in snap['seed']))
            elif name == 'section': m = 'spore #%d: section built, %d points across the flow' % (k, int(np.prod(np.shape(snap['section'])[:-1])))
            elif name == 'row': m = 'spore #%d: row %s added (rows %d+%d)' % (k, s._dir(snap['dir']), c['nb'], c['nf'])
            elif name == 'side': m = 'spore #%d: grew sideways %s (width %s)' % (k, s._dir(snap['dir']), np.round(c['r'], 3).tolist())
            elif name == 'stop': m = 'spore #%d: %s stopped: %s' % (k, s._dir(snap['dir']), snap['reason'])
            elif name == 'reject': m = ('spore #%d rejected: %s' % (k, snap['reason'])) if s._was_in else 'seed rejected: %s' % snap['reason']
            elif name == 'cell': m = 'spore #%d done: rows %d+%d, r=%s' % (k, c['nb'], c['nf'], np.round(c['r'], 3).tolist())
            else: return
            s.echo(m)
        except KeyError: pass                                       # пауза не из growN (нет dir/cell) — не наша
        except Exception as e: s.echo('[stepper] log error: %r' % (e,))

    # --- главный поток ---
    def cmd(s, key, ref=None):
        """'n' | 'm' | 'c'. Возвращает False, если алгоритм не стоит на паузе (команда проигнорирована)."""
        if s.finished or not s._stopped.is_set(): return False
        s.mode = key
        if key == 'm' and ref is not None: s.ref_lvl = ref                    # явный уровень (глубина шага)
        elif key == 'm' and s.snap: s.ref_lvl = s.snap['lvl']
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
        if s.render_n: rows.append(('render (main thread)', s.render_n, s.render_time, 1000 * s.render_time / s.render_n))
        return rows

    def abort(s):
        s._abort = True; s._go.set()
        if s._th: s._th.join(2.)
