"""Сводная статистика слоёв v8 (утилита, не тест): для каждого слоя из g.ULAYERS и посевов rng 0..4 — клетки, узлы, покрытие, нахлёст, размеры, Δτ, причины стопов, reject'ы, время.
Печатает таблицу и пишет reports/stats_layers.md. Плюс одна строка сравнения с коммитом BASE (growN до правок сессии; env как тогда в live.py: MAXC 60, NFAIL 40, без GM/зацепа/NF) — если файл достаётся из git.
Запуск (Windows-питон с numpy): python tests/stats_layers.py   |   внутреннее: python tests/stats_layers.py --json <модуль>"""
import os, sys, json, time, collections, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
BASE = '3d96972'; SEEDS = range(5); NPROBE = 5000
ENV = dict(SYS='di', M='3', KF='21', MAXC='60', NFAIL='400', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000', GS='0', GLIM='0', GSEED='1', GOALB='0', GOALSHAPE='ball', RHO='.2')   # как live.py


def run_json(modname):
    """слои × посевы в этом процессе → список словарей метрик"""
    import numpy as np, importlib
    for k, v in ENV.items(): os.environ.setdefault(k, v)
    g = importlib.import_module('src.algo.' + modname); out = []
    layers = getattr(g, 'ULAYERS', (g.US[1],)); slow = getattr(g, 'slowf', lambda y, u: np.zeros(np.shape(y)[:-1], bool))
    area = float(np.prod(2 * g.XLV))
    for u in layers:
        for sd in SEEDS:
            rej = collections.Counter()
            def pause(ph, lvl, **k):
                if ph == 'reject': rej[k['reason']] += 1
            g.pause = pause; st0 = collections.Counter(g.STOP)
            t = time.perf_counter(); cells, idx = g.build_layer(u, np.random.default_rng(sd)); t = time.perf_counter() - t
            stops = collections.Counter(g.STOP); stops.subtract(st0)
            rp = np.random.default_rng(100 + sd); pr = np.array([q for q in (g.rand_seed(rp) for _ in range(NPROBE)) if g.inbox(q) and not g.ingoal(q) and not slow(q, u)])
            cov = idx.covered(g.wrapy(pr)); ins = np.zeros(len(pr), bool); ins[idx.query(g.wrapy(pr))[0]] = True
            # нахлёст по площади ядра: сетка 9×9 внутри ядра новой клетки, доля точек в ядрах прежних
            idx2 = g.HexIdx(); wt = wc = 0.; nodes = nodes_h = 0; dm = []
            for c in cells:
                hb, hf = getattr(c, 'hb', 0), getattr(c, 'hf', 0); nr = c.nb + c.nf + 1; M = c.G.shape[1]; nodes += nr * M ** (g.m_); nodes_h += c.G.shape[0] * M ** (g.m_)
                if hasattr(c, 'tau') and nr > 1: dm.append(float(np.abs(np.diff(c.tau[hb:hb + nr], axis=0)).min() / g.DTN))
                if idx2.n:
                    G = c.G; mid = M // 2; ri = np.round(hb + (np.linspace(-.95, .95, 9) + 1) / 2 * (nr - 1)).astype(int); P = []
                    for i in ri:
                        for b in np.linspace(-.95, .95, 9):
                            e = G[i, -1] if b >= 0 else G[i, 0]; P.append(G[i, mid] + abs(b) * (e - G[i, mid]) / (1 + g.HALO))
                    m = idx2.covered(np.array(P)); ar = c.r[0] * nr; wt += ar; wc += ar * m.mean()
                idx2.add(c)
            err = 0.; lx = []                                                                              # ошибка линейной интерполяции по времени на оси (доли RMAX) и длина клеток по x в зоне |v|<.5
            for c in cells:
                if hasattr(c, 'Gt') and len(c.tt) > 1:
                    Gc = c.Gt[:, c.Gt.shape[1] // 2]
                    for k in range(len(Gc) - 1):
                        hh = c.tt[k + 1] - c.tt[k]; mid = g.rk4(Gc[k], u, hh / 4, 2); err = max(err, float(np.linalg.norm(mid - (Gc[k] + Gc[k + 1]) / 2)) / g.RMAX)
                Gm = c.G[:, c.G.shape[1] // 2]
                if abs(c.c[1]) < .5: lx.append(float(Gm[:, 0].max() - Gm[:, 0].min()))
            r = np.array([c.r[0] for c in cells]); rows = np.array([c.nb + c.nf for c in cells])
            out.append(dict(u=u, seed=sd, cells=len(cells), nodes=nodes, nodes_h=nodes_h, area_per_node=area / max(nodes, 1), unc_core=1 - float(cov.mean()), unc_halo=1 - float(ins.mean()), overlap=wc / wt if wt else 0.,
                            r_med=float(np.median(r)), r_min=float(r.min()), r_max=float(r.max()), rows_med=float(np.median(rows)), rows_min=int(rows.min()), rows_max=int(rows.max()),
                            dtau_med=float(np.median(dm)) if dm else None, dtau_min=float(min(dm)) if dm else None, stops={k: v for k, v in stops.items() if v}, rej=dict(rej), err=err, lx=float(np.median(lx)) if lx else None, ms=1e3 * t, ms_cell=1e3 * t / max(len(cells), 1)))
    return out


def sub(mod, extra=None):
    env = dict(os.environ); env.update(ENV); env.update(extra or {}); env['PYTHONIOENCODING'] = 'utf-8'
    if mod != 'growN': env.update(MAXC='60', NFAIL='40')
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--json', mod], cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    for ln in p.stdout.splitlines()[::-1]:
        if ln.startswith('JSON '): return json.loads(ln[5:])
    raise RuntimeError(p.stderr[-800:] or p.stdout[-800:])


def ms(vals, f='%.3g'):
    import numpy as np
    vals = [v for v in vals if v is not None]
    return '–' if not vals else (f + ' ± ' + f) % (np.mean(vals), np.std(vals))


def main():
    import numpy as np
    cur = sub('growN'); lines = []
    layers = sorted({r['u'] for r in cur}, key=lambda x: [1., -1., 0.].index(x) if x in (1., -1., 0.) else 9)
    rows = [('клеток', 'cells', '%.1f'), ('узлов без гало-строк', 'nodes', '%.0f'), ('узлов с гало-строками', 'nodes_h', '%.0f'), ('площадь поля на узел', 'area_per_node', '%.4f'),
            ('непокрыто, ядро', 'unc_core', '%.4f'), ('непокрыто, ядро∪гало', 'unc_halo', '%.4f'), ('нахлёст по площади', 'overlap', '%.3f'),
            ('r медиана', 'r_med', '%.3f'), ('r мин', 'r_min', '%.3f'), ('r макс', 'r_max', '%.3f'), ('строк медиана', 'rows_med', '%.1f'), ('строк мин', 'rows_min', '%.0f'), ('строк макс', 'rows_max', '%.0f'),
            ('min Δτ/DTN, медиана по клеткам', 'dtau_med', '%.2f'), ('min Δτ/DTN, минимум', 'dtau_min', '%.2f'), ('ошибка интерполяции по времени на оси, макс (доли RMAX)', 'err', '%.4f'), ('длина клеток по x, медиана при |v|<.5', 'lx', '%.2f'), ('время build_layer, мс', 'ms', '%.0f'), ('мс на клетку', 'ms_cell', '%.0f')]
    lines += ['# Статистика слоёв v8 (HEAD, env как live.py, посевы rng 0..4; среднее ± разброс)', '', '| метрика | ' + ' | '.join('u=%g' % u for u in layers) + ' |', '|---|' + '---|' * len(layers)]
    for name, key, f in rows: lines.append('| %s | ' % name + ' | '.join(ms([r[key] for r in cur if r['u'] == u], f) for u in layers) + ' |')
    for title, key in (('причины стопов (сумма за 5 посевов)', 'stops'), ('reject-и затравок (сумма за 5 посевов)', 'rej')):
        lines += ['', '**%s**' % title, '']
        for u in layers:
            c = collections.Counter()
            for r in cur:
                if r['u'] == u: c.update(r[key])
            lines.append('- u=%g: ' % u + ', '.join('%s %d' % kv for kv in c.most_common()))
    tot = collections.defaultdict(lambda: [0, 0, 0.])
    for sd in SEEDS:
        rs = [r for r in cur if r['seed'] == sd]; tot[sd] = [sum(r['cells'] for r in rs), sum(r['nodes'] for r in rs), sum(r['ms'] for r in rs)]
    lines += ['', '**Итого по трём слоям вместе (на посев)**: клеток %s, узлов %s, время %s мс' % (ms([v[0] for v in tot.values()], '%.1f'), ms([v[1] for v in tot.values()], '%.0f'), ms([v[2] for v in tot.values()], '%.0f'))]
    try:
        gp = os.path.join(ROOT, 'src', 'algo', 'growN_base.py'); src = subprocess.run(['git', 'show', '%s:spores_2/v8/src/algo/growN.py' % BASE], cwd=ROOT, capture_output=True, encoding='utf-8', check=True).stdout
        open(gp, 'w', encoding='utf-8', newline='\n').write(src)
        try: b = sub('growN_base')
        finally: os.remove(gp)
        lines += ['', '**Сравнение с началом сессии** (%s, MAXC 60, NFAIL 40, слой u=+1, 5 посевов): клеток %s, узлов %s, непокрыто ядро %s (ядро∪гало %s), время %s мс; сейчас u=+1: клеток %s, узлов %s, непокрыто ядро %s, время %s мс'
                  % (BASE, ms([r['cells'] for r in b], '%.1f'), ms([r['nodes'] for r in b], '%.0f'), ms([r['unc_core'] for r in b], '%.3f'), ms([r['unc_halo'] for r in b], '%.3f'), ms([r['ms'] for r in b], '%.0f'),
                     ms([r['cells'] for r in cur if r['u'] == 1.], '%.1f'), ms([r['nodes'] for r in cur if r['u'] == 1.], '%.0f'), ms([r['unc_core'] for r in cur if r['u'] == 1.], '%.3f'), ms([r['ms'] for r in cur if r['u'] == 1.], '%.0f'))]
    except Exception as e: lines += ['', 'Сравнение с %s пропущено: %r' % (BASE, e)]
    if '--adapt' in sys.argv:
        lines += ['', '**Адаптивный шаг строки (клеток / узлов / непокрыто ядро / время мс / макс ошибка интерп.), посевы rng 0..4**', '', '| вариант | ' + ' | '.join('u=%g' % u for u in layers) + ' |', '|---|' + '---|' * len(layers)]
        for name, ex in (('ADAPT=0 (шаг DTN)', dict(ADAPT='0')), ('ETOL .01', dict(ADAPT='1', ETOL='.01')), ('ETOL .003', dict(ADAPT='1', ETOL='.003'))):
            v = sub('growN', ex); lines.append('| %s | ' % name + ' | '.join('%s / %s / %s / %s / %s' % (ms([r['cells'] for r in v if r['u'] == u], '%.1f'), ms([r['nodes'] for r in v if r['u'] == u], '%.0f'), ms([r['unc_core'] for r in v if r['u'] == u], '%.3f'), ms([r['ms'] for r in v if r['u'] == u], '%.0f'), ms([r['err'] for r in v if r['u'] == u], '%.3f')) for u in layers) + ' |')
    txt = '\n'.join(lines) + '\n'; print(txt); os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True); open(os.path.join(ROOT, 'reports', 'stats_layers.md'), 'w', encoding='utf-8', newline='\n').write(txt)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--json': print('JSON ' + json.dumps(run_json(sys.argv[2])))
    else: main()
