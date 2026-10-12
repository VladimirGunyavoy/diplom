"""Журнал запуска окна v8 (слово пользователя 2026-10-12): на каждый запуск — папка v8/runs/<ГГГГММДД-ЧЧММСС>/ с отдельными файлами (описание — runs/README.md).
Чистый Python/numpy (без Ursina, тестируется отдельно). Запись — в фоновом потоке (очередь + периодический flush), главный поток только кладёт задания в очередь.
Эталон T_ref в журнал не пишется (медленно)."""
import os, sys, re, json, time, queue, atexit, threading, subprocess, platform
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.abspath(os.path.join(HERE, '..'))
RE_COV = re.compile(r'layer\s+(\S+)\s+стоп по покрытию:\s*(\d+)\s+клеток,\s+непокрыто\s+([0-9.eE+-]+)')
RE_OVL = re.compile(r'seed u=(\S+) cells (\d+) overlap\(last (\d+)\) ([0-9.eE+-]+)')
KEYS_SKIP = {'w', 'a', 's', 'd', 'space', 'shift', 'left shift', 'right shift', 'control', 'left control', 'right control', 'alt', 'left alt', 'right alt', 'mouse move', 'escape'}
ENV_KEYS = ('SYS M KF MAXC NFAIL DELTA RMAX TMAX TQDM_MI GS GLIM GSEED GOALB GOALSHAPE RHO ZSOFT ZSOFTLA ETOL XL VL GMX WS UM COVTOL COVP SEEDW DTN HALOT GM').split()


def _jsonable(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (tuple, list)): return [_jsonable(y) for y in x]
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (int, float, str, bool)) or x is None: return x
    return repr(x)


class _Tee:
    """sys.stdout-обёртка: пишет и в исходный поток, и в grow.log (буфер, flush делает фоновый поток)"""
    _lk = threading.Lock()                                  # print из потока расчёта и из кадра окна одновременно: без замка TextIOWrapper падал (io.UnsupportedOperation: not writable)
    def __init__(s, orig, rl): s._o, s._rl = orig, rl
    def write(s, t):
        with s._lk: n = s._o.write(t); s._rl._grow_write(t); return n
    def flush(s): s._o.flush()
    def __getattr__(s, k): return getattr(s._o, k)


class RunLog:
    def __init__(s, root=None, tee=True, stamp=None):
        s.root = root or os.path.join(V8, 'runs'); os.makedirs(s.root, exist_ok=True); s.t0 = time.time(); stamp = stamp or datetime.now().strftime('%Y%m%d-%H%M%S'); s.dir = os.path.join(s.root, stamp); k = 1
        while os.path.exists(s.dir): k += 1; s.dir = os.path.join(s.root, '%s-%d' % (stamp, k))
        os.makedirs(s.dir); s.name = os.path.basename(s.dir); s._q = queue.Queue(); s._lock = threading.Lock(); s._grow = open(os.path.join(s.dir, 'grow.log'), 'a', encoding='utf-8', buffering=1 << 14); s._keys = open(os.path.join(s.dir, 'keys.log'), 'a', encoding='utf-8', buffering=1 << 12)
        s.atlas_d = {'layers': {}}; s._asig = None; s.summary = dict(run=s.name, dir=os.path.relpath(s.dir, V8).replace(os.sep, '/'), started=datetime.fromtimestamp(s.t0).isoformat(timespec='seconds'), n_vcalc=0, trajs=[], atlas={}, keys=0); s._wbuf = ''; s._stop = False
        s._th = threading.Thread(target=s._loop, daemon=True); s._th.start(); atexit.register(s.close)
        if tee: s._orig = sys.stdout; sys.stdout = _Tee(sys.stdout, s)
        s._write_summary()

    # ---- фон ----
    def _loop(s):
        while not s._stop or not s._q.empty():
            try: fn = s._q.get(timeout=1.)
            except queue.Empty: fn = None
            try:
                if fn is not None: fn()
                with s._lock: s._grow.flush(); s._keys.flush()
            except Exception as e: sys.__stderr__.write('[runlog] %r\n' % (e,))

    def _put(s, fn): s._q.put(fn)

    def close(s):
        if s._stop: return
        s.summary['finished'] = datetime.now().isoformat(timespec='seconds'); s._write_summary(); s._stop = True; s._th.join(5.)
        try: s._grow.flush(); s._keys.flush(); s._grow.close(); s._keys.close()
        except Exception: pass

    def _wjson(s, name, obj):
        p = os.path.join(s.dir, name); tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(_jsonable(obj), f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)

    def _append(s, name, line):
        with open(os.path.join(s.dir, name), 'a', encoding='utf-8') as f: f.write(line + '\n')

    # ---- params.json ----
    def params(s, extra=None):
        """все параметры: growN (верхний регистр), env, solve8/vsurf, слои, git, время старта"""
        def run():
            d = dict(run=s.name, started=s.summary['started'], argv=sys.argv, python=platform.python_version(), host=platform.node(), cwd=os.getcwd())
            try: d['git'] = dict(commit=subprocess.check_output(['git', '-C', V8, 'rev-parse', 'HEAD'], timeout=5, text=True).strip(), dirty=bool(subprocess.check_output(['git', '-C', V8, 'status', '--porcelain', '--', '.'], timeout=5, text=True).strip()))
            except Exception as e: d['git'] = repr(e)
            d['env'] = {k: v for k, v in sorted(os.environ.items()) if k in ENV_KEYS or k.startswith('VSURF_')}
            try:
                from src.algo import growN as g; d['growN'] = {k: _jsonable(v) for k, v in vars(g).items() if k.isupper() and not k.startswith('_') and isinstance(v, (int, float, str, bool, tuple, list, np.ndarray, np.floating, np.integer))}
            except Exception as e: d['growN'] = repr(e)
            try: from src.stepper import vsurf as VS; d['vsurf'] = {k: _jsonable(v) for k, v in vars(VS).items() if k.isupper() and isinstance(v, (int, float, str, bool, dict, np.ndarray))}
            except Exception as e: d['vsurf'] = repr(e)
            try: from src.algo import solve8 as S; d['solve8'] = dict(POLICIES=_jsonable(S.POLICIES), BIG=S.BIG)
            except Exception as e: d['solve8'] = repr(e)
            if extra: d.update(_jsonable(extra))
            s._wjson('params.json', d); s.summary['params'] = {k: d['growN'].get(k) for k in ('ZSOFT', 'ZSOFTLA', 'GM', 'GMX', 'XL', 'VL', 'M', 'RHO', 'KF', 'DELTA') if isinstance(d['growN'], dict)}; s.summary['git'] = d['git']; s._write_summary()
        s._put(run)

    # ---- grow.log + разбор строк покрытия/нахлёста ----
    def _grow_write(s, t):
        with s._lock: s._grow.write(t)
        s._wbuf += t                                                                       # print пишет кусками (аргумент за аргументом) — разбираем по законченным строкам
        if '\n' not in s._wbuf: return
        *lines, s._wbuf = s._wbuf.split('\n')
        for line in lines:
            if 'стоп по покрытию' not in line and 'overlap(' not in line: continue
            m = RE_COV.search(line)
            if m: s.atlas_d['layers'].setdefault(m.group(1), {}).update(cells_final=int(m.group(2)), uncovered=float(m.group(3)), t_done_s=round(time.time() - s.t0, 1))
            m = RE_OVL.search(line)
            if m: s.atlas_d['layers'].setdefault(m.group(1), {}).update(overlap_last=float(m.group(4)), overlap_window=int(m.group(3)))

    # ---- atlas.json ----
    def atlas(s, snap):
        """по слоям: клеток, узлов; покрытие и нахлёст — из строк роста; пересохраняется при изменении набора клеток"""
        if not snap: return
        cbl = dict(snap.get('cells_by_layer') or {}); layers = snap.get('layers') or (); cur = snap.get('layer_k', 0)
        if layers and cur < len(layers) and snap.get('cells'): cbl[layers[cur]] = snap['cells']
        sig = tuple((str(u), len(c), id(c[-1]['G']) if c else 0) for u, c in cbl.items())
        if sig == s._asig: return
        s._asig = sig; info = {}
        for u, cl in cbl.items():
            nt = sum(int(np.shape(c['G'])[0]) for c in cl); M = int(np.shape(cl[0]['G'])[1]) if cl else 0; info[str(float(u))] = dict(cells=len(cl), rows=nt, nodes=nt * M, M=M)
        def run():
            L = s.atlas_d['layers']
            for k, v in info.items(): L.setdefault(k, {}).update(v)
            s.atlas_d.update(phase=snap.get('phase'), updated_s=round(time.time() - s.t0, 1)); s._wjson('atlas.json', s.atlas_d); s.summary['atlas'] = {k: {x: v.get(x) for x in ('cells', 'nodes', 'uncovered', 'overlap_last')} for k, v in L.items()}; s._write_summary()
        s._put(run)

    # ---- vcalc.jsonl ----
    def vcalc(s, mode, info, cells, solve_s, cache='miss'):
        row = dict(t_s=round(time.time() - s.t0, 2), mode=mode, cache=cache, cells=int(cells), layers=_jsonable(info.get('layers')), nodes=int(info.get('n', 0)), passes=int(info.get('passes', 0)), converged=bool(info.get('converged', False)), solve_s=round(float(info.get('t', solve_s)), 3), total_s=round(float(solve_s), 3), finite=round(float(info.get('finite', 0.)), 4), vmax=round(float(info.get('vmax', 0.)), 4))
        def run(): s._append('vcalc.jsonl', json.dumps(row, ensure_ascii=False)); s.summary['n_vcalc'] += cache == 'miss'; s.summary['n_vcalc_hits'] = s.summary.get('n_vcalc_hits', 0) + (cache == 'hit'); s.summary['last_vcalc'] = row; s._write_summary()
        s._put(run)

    # ---- traj_<режим>.csv + traj_summary.jsonl ----
    @staticmethod
    def _tdiag(res, i, status, T):
        """диагностика траектории (почему занижение/фейл): T_ref, T/T_ref, V*(x0), V*(x0)−T_ref (<0 — занижение в старте), max рост V* вдоль пути (горб), V* и скорость в конце, u последнего шага, число точек, причина (reason)"""
        nan = float('nan'); Tr = float(res['Tref'][i]) if 'Tref' in res else nan; V = np.asarray(res['V'][i], float); P = np.asarray(res['paths'][i], float); U = np.asarray(res['U'][i], float)
        fin = V[np.isfinite(V) & (V < 1e8)]; V0 = float(fin[0]) if len(fin) else nan; Ve = float(V[-1]) if len(V) and np.isfinite(V[-1]) and V[-1] < 1e8 else nan
        d = np.diff(np.where(np.isfinite(V) & (V < 1e8), V, nan)) if len(V) > 1 else np.zeros(0); hump = float(np.nanmax(d)) if np.isfinite(d).any() else nan
        k = max(len(P) // 5, 2); span = float(np.linalg.norm(P[-1][:2] - P[-k][:2])) if len(P) > 2 else nan; ul = float(U[-1]) if len(U) else nan
        if status == 'reached': reason = 'ok'
        elif status == 'out_of_area': reason = 'start_out_of_V_area'
        elif not np.isfinite(Ve) or (len(V) >= 5 and (~(np.isfinite(V[-5:]) & (V[-5:] < 1e8))).sum() >= 3): reason = 'left_V_area_or_inf_at_end'      # V* конца бывает конечен случайно (узкая полоска у шва) — смотрим на последние 5 точек
        elif span < 0.02: reason = 'stalled_at_end'
        else: reason = 'timeout_or_cycle'
        return (Tr, T / Tr if np.isfinite(T) and np.isfinite(Tr) and Tr > 0 else nan, V0, V0 - Tr if np.isfinite(V0) else nan, hump, Ve, span, ul, len(P), reason)

    def traj(s, mode, res, seconds=None):
        """таблица стартов: x0, v0, status (reached / failed / out_of_area), left_work_area, T, переключений, доля записей по u=+1/−1/0, конечная точка у недошедших; сводка — по траекториям не вне области (out_of_area — отдельным числом) и отдельно по оставшимся целиком в ±XL"""
        T = np.asarray(res['T'], float); n = len(T); st = np.asarray(res['status'], object); left = np.asarray(res['left_work'], bool); pre = np.asarray(res['prephase'], bool); nan = float('nan'); rows = []
        for i in range(n):
            U = np.asarray(res['U'][i], float)[1:]; tot = max(len(U), 1); sh = [float((U > 0).sum()) / tot, float((U < 0).sum()) / tot, float((U == 0).sum()) / tot]; x0 = np.asarray(res['X0'][i], float)[:2]; end = res['paths'][i][-1][:2] if st[i] != 'reached' else (nan, nan)
            rows.append((i, x0[0], x0[1], str(st[i]), int(left[i]), int(pre[i]), T[i], int(res['sw'][i]), *sh, *end, *s._tdiag(res, i, st[i], T[i])))
        ok = st == 'reached'; nin = st != 'out_of_area'; w = nin & ~left; okw = ok & ~left; sw_ = np.asarray(res['sw'], float); q = lambda a, p: float(np.percentile(a, p)) if len(a) else None
        summ = dict(t_s=round(time.time() - s.t0, 1), mode=mode, n=int(n), out_of_area=int((~nin).sum()), in_area=int(nin.sum()), prephase_starts=int(pre.sum()), reached=int(ok.sum()), failed=int((nin & ~ok).sum()), T_median=q(T[ok], 50), T_p90=q(T[ok], 90), switches_mean=float(sw_[nin].mean()) if nin.any() else None,
                    work=dict(n=int(w.sum()), reached=int(okw.sum()), failed=int((w & ~ok).sum()), T_median=q(T[okw], 50), T_p90=q(T[okw], 90), switches_mean=float(sw_[w].mean()) if w.any() else None), left_work_area=int(left.sum()), seconds=None if seconds is None else round(float(seconds), 1))
        def run():
            with open(os.path.join(s.dir, 'traj_%s.csv' % mode), 'w', encoding='utf-8') as f:
                f.write('i,x0,v0,status,left_work_area,prephase,T,switches,share_u+1,share_u-1,share_u0,end_x,end_v,T_ref,T_over_Tref,V0,V0_minus_Tref,max_V_rise,V_end,end_span,last_u,n_pts,reason\n')
                for r in rows: f.write('%d,%.6g,%.6g,%s,%d,%d,%.6g,%d,%s\n' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], ','.join('%.4f' % x for x in r[8:11]) + ',' + ','.join('%.6g' % x for x in r[11:13]) + ',' + ','.join('%.6g' % x for x in r[13:21]) + ',%d,%s' % (r[21], r[22])))
            dec = res.get('decisions')
            if dec is not None and len(dec):                                                          # журнал решений агента по траекториям (b2: run_agent(dlog=...)): одна строка на решение
                with open(os.path.join(s.dir, 'traj_decisions.csv'), 'w', encoding='utf-8') as f:
                    f.write(','.join(res.get('decisions_cols') or []) + '\n')
                    for r in np.asarray(dec, object): f.write(','.join(('%.6g' % x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + '\n')
            s._append('traj_summary.jsonl', json.dumps(summ, ensure_ascii=False)); s.summary['trajs'] = [t for t in s.summary['trajs'] if t['mode'] != mode] + [summ]; s._write_summary()
        s._put(run)

    # ---- keys.log ----
    def key(s, key, state=None, extra=None):
        if key in KEYS_SKIP or key.endswith(' up') or ' hold' in key or key.startswith('scroll'): return
        line = '%.2f\t%s\t%s\n' % (time.time() - s.t0, key, json.dumps(_jsonable(dict(state or {}, **(extra or {}))), ensure_ascii=False) if (state is not None or extra) else '')
        with s._lock: s._keys.write(line)
        s.summary['keys'] += 1

    # ---- summary.json + общий index.jsonl ----
    def _write_summary(s):
        def run():
            s._wjson('summary.json', s.summary); rows = []
            for d in sorted(os.listdir(s.root)):
                p = os.path.join(s.root, d, 'summary.json')
                if os.path.isfile(p):
                    try: rows.append(json.load(open(p, encoding='utf-8')))
                    except Exception: pass
            tmp = os.path.join(s.root, 'index.jsonl.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
            os.replace(tmp, os.path.join(s.root, 'index.jsonl'))
        if getattr(s, '_sw_pending', False): return
        s._sw_pending = True
        def deb(): s._sw_pending = False; run()
        s._q.put(deb)
