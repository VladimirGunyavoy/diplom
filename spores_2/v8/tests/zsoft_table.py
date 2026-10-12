"""Сводка свипа ZSOFT (утилита): python tests/zsoft_table.py <папка_с_ov_/st_/s2_/s3_> <z1> <z2> … → таблицы markdown (заезд, слои, solve8)"""
import json, re, sys
D = sys.argv[1]; Z = sys.argv[2:]
print('| ZSOFT | ядро клетки 1 в ядрах 2+3, % (u=+1 ; u=−1) | ядро 1 в ядре+гало 2+3, % | ядро+гало кл.2 / кл.3 в ядре+гало прежних, % (u=+1) | нахлёст слоя u=+1/−1/0 | клеток u=+1/−1/0 | непокрыто ядро u=+1/−1/0 | непокрыто ядро∪гало u=+1/−1/0 |'); print('|' + '---|' * 8)
for z in Z:
    R = json.load(open('%s/ov_%s.json' % (D, z))); a = []; b = []; c = []
    for r in R:
        P = {(q['i'], q['j']): q for q in r['pairs']}; a.append('%.0f+%.0f' % (100 * P[(1, 2)]['core_in_core'], 100 * P[(1, 3)]['core_in_core'])); b.append('%.0f+%.0f' % (100 * P[(1, 2)]['core_in_all'], 100 * P[(1, 3)]['core_in_all'])); c.append('%.0f / %.0f' % (100 * r['cells'][1]['full_by_prev_all'], 100 * r['cells'][2]['full_by_prev_all']))
    st = {}
    for l in open('%s/st_%s.md' % (D, z), encoding='utf-8'):
        m = re.match(r'\| (клеток|непокрыто, ядро|непокрыто, ядро∪гало|нахлёст по площади) \| (.*?) \| (.*?) \| (.*?) \|', l)
        if m: st[m.group(1)] = [re.sub(r' ±.*', '', x) for x in m.groups()[1:]]
    f = lambda k: '/'.join(st[k])
    print('| %s | %s | %s | %s | %s | %s | %s | %s |' % (z, ' ; '.join(a), ' ; '.join(b), c[0], f('нахлёст по площади'), f('клеток'), f('непокрыто, ядро'), f('непокрыто, ядро∪гало')))
print(); print('| ZSOFT | слои | E3 | политика | дошли (все) | дошли (оптимум в ±3.5) | T/T_ref мед / p90 (±3.5) | переключений |'); print('|' + '---|' * 8)
for z in Z:
    for L, nm in (('s2', '±1'), ('s3', '±1,0')):
        try: R = json.load(open('%s/%s_%s.json' % (D, L, z)))
        except Exception as e: print('|', z, nm, 'нет', e, '|'); continue
        for mode, r in R.items():
            for pn in ('base', 'look'):
                p = r['policies'][pn]; print('| %s | %s | %s | %s | %.3f | %.3f | %.2f / %.2f | %.0f |' % (z, nm, mode, pn, p['success'], p['in35']['success'], p['in35']['T_over_ref_median'], p['in35']['p90'], p['in35']['switches_mean']))
