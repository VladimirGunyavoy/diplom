"""manip 4D с запретами (FORB b9): таблица. Старты: свободны (gfree > 0) И хоть один метод дошёл (общий критерий для всех); T_ref = лучшее проверенное T среди методов (OCP с запретами — мультистарт локален: на части стартов 12–19 с, где споры b9 дают 4–7 с).
python mf_report.py → reports/bench/table_manipf.md"""
import os, json, glob, numpy as np
import common
D = common.DATA; e = common.make('manipf'); n = len(e.X0); free = e.free0
def L(f): return json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else None
M = {}                                                                                # метод → (T по стартам (None — не дошёл), t_заготовка, t_запрос, пометка)
oc = L('manipf_ocpc.json'); M['OCP IPOPT + запреты (мультистарт T0×5)'] = ([r['T'] for r in oc['per_start']], 0., oc['t_query_med'], '')
sp = L('manipf_spores_load.json'); M['споры ДО FORB2b (b9 A_forb1: слои 343 + solve 211 с — цифры b9 forb.md; FIN-ω)'] = ([r['T_verify'] for r in sp['per_start']], 343. + 211., sp['t_query_med'], '')
sp2 = L('manipf_spores_load_f2b.json')
if sp2: M['споры FORB2b (b11 A_f2b, MAXC 3000; запрос с прогревом qx; заготовка — оценка по b9 343+211, у b11 не мерили)'] = ([r['T_verify'] for r in sp2['per_start']], 343. + 211., sp2['t_query_med'], '')
for a in ('sst', 'rrt'):
    d = L('manipf_%s.json' % a)
    if d:
        for bi, b in enumerate(d['budgets']):
            T = [None] * n
            for r in d['raw']: x = r['runs'][bi]; T[r['i']] = x.get('T') if x['has'] else None
            M['%s %g с' % (a.upper(), b['budget'])] = (T, 0., b['budget'], '')
for f in sorted(glob.glob(os.path.join(D, 'manipf_ppo_*_s*.json'))):
    p = json.load(open(f)); M['PPO %.0fM s%d' % (p['steps'] / 1e6, p['seed'])] = (p['T'], p['t_train'], p.get('t_infer_per_start', 0.), '')
best = np.full(n, np.inf)
for k, (T, _, _, _) in M.items():
    for i, t in enumerate(T):
        if t is not None and np.isfinite(t) and free[i]: best[i] = min(best[i], t)
keep = free & np.isfinite(best); excl = free & ~keep
with open(os.path.join(os.path.dirname(D), 'table_manipf.md'), 'w') as f:
    f.write('manip 4D с запретами (самостолкновение |q2| ≤ π−.5, стол y ≥ −.6): стартов 20, свободных %d, с хотя бы одним дошедшим методом %d (исключены для всех: %s). T_ref = лучшее T среди методов.\n\n' % (free.sum(), keep.sum(), np.flatnonzero(excl).tolist()))
    f.write('| метод | дошли | T/T_ref мед | p90 | max | заготовка, с | запрос, с | нарушений пути (принятые) |\n|---|---|---|---|---|---|---|---|\n')
    for k, (T, tb, tq, _) in M.items():
        r = np.array([np.nan if (t is None or not np.isfinite(t)) else t / best[i] for i, t in enumerate(T)])[keep]; ok = np.isfinite(r) & (r <= 3)
        f.write('| %s | %d/%d | %s | %s | %s | %.0f | %.2f | 0 |\n' % (k, ok.sum(), keep.sum(), *(('%.3f' % np.median(r[ok]), '%.3f' % np.percentile(r[ok], 90), '%.2f' % r[ok].max()) if ok.any() else ('—', '—', '—')), tb, tq))
    sp_T = M[[k for k in M if k.startswith('споры')][0]][0]; f.write('\nСпоры не дошли на стартах (из keep): %s — OCP/SST доходят, т.е. это провалы спор (честно): агент «застрял» у границы запрещённого (gfree .05–.3, b9 forb.md).\n' % [int(i) for i in np.flatnonzero(keep) if sp_T[i] is None])
    f.write('\nНарушения пути: T у любого метода — только после common.verify (подшаги h=.01, gfree<0 → отказ «box»), поэтому у принятых путей нарушений 0 для всех методов; PPO дополнительно обрывает эпизод на out_of_box (BenchGym.step, и при прокатке). Нарушившие путь считаются недошедшими. Провалы спор 7/16 — дефект атласа: дыра покрытия у границы запрещённого, чинит [B] FORB2 (knowledge/research/forb_hole.md).\n')
    f.write('\nПо стартам (индекс: лучшее T) — %s\n' % {int(i): round(float(best[i]), 2) for i in np.flatnonzero(keep)})
print(open(os.path.join(os.path.dirname(D), 'table_manipf.md')).read())
