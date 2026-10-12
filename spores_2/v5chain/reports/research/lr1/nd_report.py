"""nd_results.json → nd_results.md (раздел «nD» для lr1/results.md). python nd_report.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, 'nd_results.json')))
R.sort(key=lambda x: (x['SYS'] != 'dd', x['u'][0] != 1, x['seed']))
L = ['## nD (worker-1: SYS=dd слои (1,0), (.5,.5); SYS=manip 4D)', '',
     'Код: `nd_qn.py` (динамика — `import growN` из v8, без правок), таблицы — `nd_report.py` из `nd_results.json`. h = .1, K = 11 узлов на ось, 15 шагов, r ∈ {.2, .5}.',
     'Последний (15-й) фронт. cos — |cos(касательная фронта, f)| (0 = ортогонально). τ-разброс: V0 = 0 по построению; V2 — «последний шаг / накоплено за 15 шагов» (max−min сдвигов δ); V3 — max−min времени прихода '
     'ближайшей траектории старых узлов (шаг интерполяции по отрезку). Ширина — поперёк потока на последнем шаге / 2r (V3 заново строит кривые длины r от оси → ≈1 по построению, сужение потока в ней не видно). '
     'f/фронт — векторных вычислений f на один фронт (без проверки согласованности), мс — wall на фронт.', '']
for d in R:
    L += ['### %s u=%s зерно %s (n=%d, m=%d, f(зерно)=%s)' % (d['SYS'], tuple(d['u']), tuple(d['seed']), d['n'], d['m'], [round(v, 3) for v in d['f_seed']]), '',
          '| r | вариант | узлов | кривых | cos mean | cos max | τ-разброс | ширина/2r | f/фронт | мс/фронт | примечание |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for x in d['runs']:
        tau = ('%.4f / %.4f' % (x['time_spread'], x['time_spread_cum'])) if x['variant'] == 'V2' else '%.4f' % x['time_spread']; note = ''
        if x['variant'] == 'V2': note = 'МНК-невязка %.3f r; разреженный lsqr, плотный lstsq %.1f мс' % (x['lsq_resid_last'], x['ms_dense'])
        if x['variant'].startswith('V3'): note = 'согл.: dist mean %.3f r / max %.3f r, <τ> %.4f' % (x['cons_dist_mean'], x['cons_dist_max'], x['time_mean'])
        L.append('| %.1f | %s | %d | %s | %.4f | %.4f | %s | %.3f | %.0f | %.2f | %s |' % (x['r'], x['variant'], x['nodes'], x.get('curves', '—'), x['cos_mean'], x['cos_max'], tau, x['width'], x['f_per_front'], x['ms_per_front'], note))
    L += ['', '**Метрика 5** (V3б против V3а: отклонение узлов комбинированных кривых от аддитивной суперпозиции одноканальных, доли r; последний фронт / по всем 15 фронтам max):', '',
          '| r | кривых а → б | mean | max | max по всем фронтам | время б / а (мс) |', '|---|---|---|---|---|---|']
    for r in sorted({x['r'] for x in d['runs']}):
        a = [x for x in d['runs'] if x['r'] == r and x['variant'] == 'V3a'][0]; b = [x for x in d['runs'] if x['r'] == r and x['variant'] == 'V3b'][0]
        L.append('| %.1f | %d → %d | %.4f | %.4f | %.4f | %.2f / %.2f |' % (r, a['curves'], b['curves'], b['comb_dev_last_mean'], b['comb_dev_last_max'], b['comb_dev_all_max'], b['ms_per_front'], a['ms_per_front']))
    L += ['', '**Клоны V3** (для каждого узла нового фронта: расстояние до ближайшей траектории от узлов СТАРОГО фронта на t ∈ [0, 2h], доли r, и время прихода τ; фронты 2..15; τ идеально = h = .1):', '',
          '| r | вариант | dist mean (по фронтам) | dist max (по всем) | dist mean: 2-й → 15-й фронт | <τ> | τ-разброс за фронт: mean / max | τ min…max (по всем) |', '|---|---|---|---|---|---|---|---|']
    for x in d['runs']:
        if x['variant'].startswith('V3'): L.append('| %.1f | %s | %.4f | %.4f | %.4f → %.4f | %.4f | %.4f / %.4f | %.4f … %.4f |' % (x['r'], x['variant'], x['cons_k_mean'], x['cons_k_max'], x['cons_first_mean'], x['cons_last_mean'], x['tau_k_mean'], x['tau_k_spread_mean'], x['tau_k_spread_max'], x['tau_min'], x['tau_max']))
    L += ['', '**Метрика 6** (голономия V3 из зерна, пути на (r, r); подшаги ds = r/50, средняя точка / Эйлер; AB = «e_a потом e_b» против «e_b потом e_a»; D = диагональ (e_a+e_b)/√2 длиной r√2; сдвиг — вдоль f в единицах времени):', '',
          '| r | интегратор | пара осей | AB: dist/r | AB: сдвиг, ед. времени | D−A: dist/r | D−A: сдвиг | D−B: dist/r | D−B: сдвиг |', '|---|---|---|---|---|---|---|---|---|']
    for h in d['holonomy']:
        for q in h['pairs']:
            L.append('| %.1f | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |' % (h['r'], 'RK2' if h['order'] == 2 else 'Эйлер', tuple(q['pair']), q['AB_dist_over_r'], q['AB_shift_t'], q['D_A_dist_over_r'], q['D_A_shift_t'], q['D_B_dist_over_r'], q['D_B_shift_t']))
    L.append('')
open(os.path.join(HERE, 'nd_results.md'), 'w', encoding='utf-8').write('\n'.join(L)); print('written', len(L), 'lines')
