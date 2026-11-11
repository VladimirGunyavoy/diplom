"""Свод R20-динамик: дерево+V (car_dyn.py, mode test) против Б-GPU/А атласа (r20_dyn_compact.json, c8), парно по (сид дисков k, старт i). Запуск: python3 agg_dyn.py <папка с car_dyn_test_*.json> <r20_dyn_compact.json>"""
import sys, glob, json, numpy as np
d, comp = sys.argv[1], json.load(open(sys.argv[2])); mine = {}
for f in glob.glob(d + '/car_dyn_test_*.json'):
    for k, v in json.load(open(f)).items(): mine[int(k)] = {r[0]: r for r in v}
rows = []   # (k, i, Tref, T_mine, T_B, T_A, touch_mine, ms)
for k in sorted(mine):
    B = {r['i']: r for r in comp[str(k)]['B']}; A = {r['i']: r for r in comp[str(k)]['A']}
    for i, r in mine[k].items():
        if i not in B: continue    # пары вне отбора c8 (внутри диска / без эталона)
        rows.append((k, i, r[2], r[1], B[i]['T'] if B[i]['ok'] else None, A[i]['T'] if A[i]['ok'] else None, r[3] < 0, r[4], bool(B[i]['touch']), bool(A[i]['touch']), B[i]['ms']))
def stat(name, T, ref):
    ok = [(t / r) for t, r in zip(T, ref) if t is not None]; a = np.array(ok)
    print('%-22s дошли %d/%d  мед %.3f p90 %.3f max %.3f' % (name, len(ok), len(T), np.median(a), np.percentile(a, 90), a.max()))
ref = [r[2] for r in rows]; print('сидов %d, пар %d' % (len(mine), len(rows)))
stat('дерево+V (TIE.1 DEP3)', [r[3] for r in rows], ref); stat('Б-GPU атласа', [r[4] for r in rows], ref); stat('А атласа', [r[5] for r in rows], ref)
print('касаний: дерево %d, Б %d, А %d; мс/шаг: дерево %.1f, Б %.1f' % (sum(r[6] for r in rows), sum(r[8] for r in rows), sum(r[9] for r in rows), 1e3 * np.mean([r[7] for r in rows]), np.mean([r[10] for r in rows])))
both = [(r[3], r[4], r[2]) for r in rows if r[3] is not None and r[4] is not None]; dt = np.array([(a - b) / c for a, b, c in both])
print('парно (дерево−Б)/T_ref по %d парам: мед %+.3f, среднее %+.3f, доля дерево быстрее %.2f' % (len(both), np.median(dt), dt.mean(), (dt < -1e-9).mean()))
