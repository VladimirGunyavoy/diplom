"""CA-MAZE: reports/bench/table_maze.md из data/maze_*.json. Споры/PRM — пересчёт по per_start с актуальным T_ref (A*+натяжение, верхняя оценка); SST/RRT/OCP/PPO — как посчитаны на aida (дошли = 0 — от T_ref не зависит)."""
import os, glob, json, numpy as np, common
D = common.DATA; e = common.make('maze'); L = lambda f: json.load(open(f'{D}/{f}')) if os.path.exists(f'{D}/{f}') else None
def f(x, n=3): return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else ('%.' + str(n) + 'g') % x
rows = []
def add(m, par, build, qm, qp, s, w, note=''): rows.append('| %s | %s | %s | %s / %s | %s | %.0f%% | %s / %s / %s | %s | %s |' % (m, par, f(build), f(qm), f(qp), f(None if build is None else build + 50 * qm), 100 * s['reached'], f(s['ratio_med']), f(s['ratio_p90']), f(s['ratio_max']), w, note))
for tag, nm in (('_b', 'FORB2B=1, 4×3000 клеток'), ('_b6', 'FORB2B=1, 4×6000 клеток'), ('_b9', 'FORB2B=1, 4×9000 клеток'), ('_a', 'FORB2B=0, 4×3000 клеток')):
    d = L(f'maze_spores{tag}.json')
    if d:
        s = common.summarize(e, [r['T_verify'] for r in d['per_start']]); ok = [r['t_query'] for r in d['per_start'] if r['T_verify']]
        add('**споры** growN dd + gfree (%s)' % nm, 'рост %s с + solve GPU %s с' % (f(d['t_grow'], 4), f(d['t_solve'], 3)), d['t_build'], float(np.median(ok)) if ok else None, float(np.percentile(ok, 90)) if ok else None, s, '%.0f МБ' % (d['weight']['bytes'] / 1e6), 'запрос — медиана по дошедшим; провал = старт не покрыт (V=∞) или тупик')
for fn in sorted(glob.glob(f'{D}/maze_prm*_*.json')):
    d = json.load(open(fn)); s = common.summarize(e, [r['T_verify'] for r in d['per_start']]); add(d['method'], 'k=15 / r(PRM*)', d['t_build'], d['t_query_med'], d['t_query_p90'], s, '%.2g МБ' % (d['weight']['bytes'] / 1e6), 'путь по (x,y), rotate–straight–rotate')
for a, nm in (('sst', 'SST'), ('rrt', 'RRT')):
    d = L(f'maze_{a}.json')
    if d:
        for b in d['budgets']: add('%s (свой), бюджет %g с' % (nm, b['budget']), 'prune %s' % d['prune'], 0., b['budget'], None, b, '—')
d = L('maze_ocp.json')
if d: add('OCP SLSQP (генерик, прямая-инициализация)', '', 0., d['t_query_med'], d['t_query_p90'], d, '—')
for fn in sorted(glob.glob(f'{D}/maze_ppo_*.json')):
    d = json.load(open(fn)); add('PPO %gM шагов (сид %d)' % (d['steps'] / 1e6, d['seed']), '', d['t_train'], d['t_infer_per_start'], None, d, '—')
H = '| метод | параметры | заготовка, с | запрос мед / p90, с | total(N=50), с | дошли | T/T_ref мед / p90 / max | вес | примечание |\n|---|---|---|---|---|---|---|---|---|\n'
open(f'{D}/../table_maze.md', 'w').write('### CA-MAZE: лабиринт 6×6 (DFS), дифдрайв 3D, 50 стартов, одна цель, T_ref = A*+натяжение (верхняя оценка, не оптимум; ratio < 1 возможен)\n\n' + H + '\n'.join(rows) + '\n')
print(open(f'{D}/../table_maze.md').read())
