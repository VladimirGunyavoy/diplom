"""CA-MAZE 4D: reports/bench/table_maze4.md из data/maze4_*.json. T_ref здесь — нижняя граница T_lb (длина кратчайшего пути / v_max = 1; недостижима: разгон, повороты), поэтому T/T_lb > 1 всегда;
сравнивать методы между собой. Доля дошедших — T ≤ 3·T_lb."""
import os, glob, json, numpy as np, common
D = common.DATA; L = lambda f: json.load(open(f'{D}/{f}')) if os.path.exists(f'{D}/{f}') else None
f = lambda x, n=3: '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else ('%.' + str(n) + 'g') % x
rows = []
CONCL = '''
**Вывод (c6, 2026-10-13 ~12:00).** В 4D-лабиринте покрытия не хватает при N клеток: 4×3000 → 0% (FORB2=.5) / 6% (FORB2=0); 4×10000 (FORB2=0 FORB2B=1 VIAB=1; рост 1758 с, solve 4688 с, 7.2M узлов) → **24% (12/50)**, T/T_lb мед 1.67, запрос .25 с; охват растёт с числом клеток, но заготовка уже 6446 с против 50 с у PRM*. Диагноз (`bench/diag_maze4.py`): цель связана (конечных V 69% узлов у цели), старты за r>1.5 не покрыты — клетки 4D (x, y, θ, v) малы и сидят у цели. SST/RRT (OMPL) — 0% при бюджете 60 с. Рабочий конкурент — PRM* по карте (x, y) + преследование пути: 90%, T/T_lb 2.2, запрос .63 с, заготовка 50 с. Итог: на MAZE 4D при доступном бюджете клеток споры проигрывают по доле дошедших и по заготовке; выигрыш по T у дошедших (1.67 против 2.2) этого не компенсирует.
'''
def add(m, par, build, qm, d, w, note=''): rows.append('| %s | %s | %s | %s | %.0f%% | %s / %s / %s | %s | %s |' % (m, par, f(build), f(qm), 100 * d['reached'], f(d['ratio_med']), f(d['ratio_p90']), f(d['ratio_max']), f(w / 1e6 if w else None), note))
for fn, nm, note in (('maze4_spores_3000.json', 'FORB2=.5, 4×3000', 'клетки у цели (r мед .29), старты не покрыты'), ('maze4_spores_f0.json', 'FORB2=0, 4×3000', 'охват ×3.4 крупнее клетки; 3/50'),
                     ('maze4_spores_f0b.json', 'FORB2=0 FORB2B=1 VIAB=1, 4×10000', '')):
    d = L(fn)
    if d: add('**споры** growN maze4 (%s)' % nm, 'рост %s с + solve %s с' % (f(d['t_grow'], 4), f(d['t_solve'], 4)), d['t_build'], d['t_query_med'], d, d['weight']['bytes'], note)
for fn in sorted(glob.glob(f'{D}/maze4_prms_*.json')):
    d = L(os.path.basename(fn)); add('PRM* + преследование пути', d['method'].split('+')[0].strip(), d['t_build'], d['t_query_med'], d, d['weight']['bytes'], 'тот же проверяльщик; запрос = путь + моделирование')
for a in ('sst', 'rrt'):
    d = L(f'maze4_{a}.json')
    if d:
        for b in d['budgets'][-2:]: add('%s (OMPL)' % a.upper(), 'бюджет %g с на запрос' % b['budget'], 0., b['budget'], b, None, '')
hdr = '| метод | параметры / заготовка | t заготовки, с | t запроса (мед), с | дошли | T/T_lb мед / p90 / max | вес, МБ | примечание |\n|---|---|---|---|---|---|---|---|\n'
open(os.path.join(D, '..', 'table_maze4.md'), 'w').write('# MAZE 4D — лабиринт 6×6, дд с инерцией (x, y, θ, v), 50 стартов с v=0\n\nСистема: ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω, v̇ = a − v, |a|+|ω| ≤ 1, |v| ≤ 1; цель — квадрат ±.08 по (x, y), θ и v любые. T_lb — нижняя граница (путь/1), см. `bench/maze4.py`.\n\n' + hdr + '\n'.join(rows) + '\n' + CONCL)
print('\n'.join(rows))
