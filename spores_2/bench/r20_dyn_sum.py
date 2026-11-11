"""Сводка R20×100 динамик кар (правило research-33): python r20_dyn_sum.py [k ...] (без аргументов — все готовые сиды) → markdown.
Класс 1: старт внутри диска при t=0 (зазор<0) — невалиден, исключён отовсюду (знаменатель = пары − такие).
Класс 2: старт валиден, OCP не сошёлся — в дошли/касания считается, из T/T_ref исключён, перечисляется отдельно."""
import os, sys, json, glob, numpy as np
T = int(os.environ.get('TIGHT', 0)); D = os.path.expanduser('~/spore_v5/c8'); DD = D + '/reports/bench/data/'; S = np.load(D + '/r20/starts100%s.npy' % ('_tight' if T else '')); PT = 'ts' if T else 's'; ST = '_tight' if T else ''
ks = list(map(int, sys.argv[1:])) or sorted(k for k in range(20) if all(os.path.exists(DD + 'env_A_random_r20%s%d_%s.json' % (PT, k, m)) for m in 'AB'))
inv = []; noref = []; acc = {m: dict(n=0, ok=0, tch=[], rt=[], ms=[], clr=[]) for m in 'AB'}; rows = []
for k in ks:
    ref = {r['i']: r for r in json.load(open(DD + 'r20_ref_s%d%s.json' % (k, ST)))}
    inv += [(k, i, round(r['clr0'], 4)) for i, r in ref.items() if r['clr0'] < 0]; noref += [(k, i) for i, r in ref.items() if r['clr0'] >= 0 and not r['T']]
    for m in 'AB':
        res = json.load(open(DD + 'env_A_random_r20%s%d_%s.json' % (PT, k, m)))['res']
        assert sorted(r['i'] for r in res) == list(range(100)) and all(r['seed'] == k and np.array_equal(np.asarray(r['Y'][0], float)[:4], S[r['i'], :4]) for r in res), (k, m)
        v = [r for r in res if ref[r['i']]['clr0'] >= 0]; a = acc[m]; ok = [r for r in v if r['ok']]; rt = [r['T'] / ref[r['i']]['T'] for r in ok if ref[r['i']]['T']]
        t = [(k, r['i']) for r in v if r['touch']]; a['n'] += len(v); a['ok'] += len(ok); a['tch'] += t; a['rt'] += rt; a['ms'] += [r['ms'] for r in v]; a['clr'] += [r['clr'] for r in v]
        rows.append('| %d | %s | %d | %d/%d | %d | %.3f | %.3f | %.3f | %.0f |' % (k, m, len(v), len(ok), len(v), len(t), np.median(rt), np.percentile(rt, 90), max(rt), np.median([r['ms'] for r in v])))
print('| сид | метод | валидных стартов | дошли | касаний | T/T_ref мед | p90 | max | мс/шаг мед |\n|---|---|---|---|---|---|---|---|---|'); print('\n'.join(rows))
for m in 'AB':
    a = acc[m]; rt = np.array(a['rt']); print('**%s итого (%d сидов, 2 цепочки)**: дошли %d/%d, касаний %d %s, T/T_ref (по %d парам) мед %.3f p90 %.3f max %.3f, мс/шаг мед %.0f, мин.зазор мед %.3f' % (m, len(ks), a['ok'], a['n'], len(a['tch']), a['tch'], len(rt), np.median(rt), np.percentile(rt, 90), rt.max(), np.median(a['ms']), np.median(a['clr'])))
print('Класс 1 (старт внутри диска, исключены): %d — %s' % (len(inv), inv)); print('Класс 2 (эталон OCP не сошёлся, в T/T_ref не входят): %d — %s' % (len(noref), noref))
